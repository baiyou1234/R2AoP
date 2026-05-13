import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.config import get_config
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D


# =========================
# Small adapter: unify model outputs
# AoP-SAM -> dict with 'masks'
# nnUNet2D -> tensor (B,C,H,W)  ==> {'masks': logits}
# =========================
def to_masks_dict(out):
    if isinstance(out, dict) and ('masks' in out):
        return out
    return {'masks': out}


# =========================
# Dice loss (for validation metric)
# We compute Dice on discrete argmax predictions (same as your original intent)
# =========================
class MyDC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        y_pred, y_truth: (N, C) one-hot, float
        """
        intersection = (
            (y_pred[:, 1:2] * y_truth[:, 1:2]).sum() +
            (y_pred[:, 2:]  * y_truth[:, 2:]).sum()
        )
        union = (
            y_pred[:, 1:2].sum() + y_pred[:, 2:].sum() +
            y_truth[:, 1:2].sum() + y_truth[:, 2:].sum()
        )
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # 1 - Dice (lower is better)


class DCloss(torch.nn.Module):
    """
    Validation metric: 1 - Dice
    Works for BOTH AoP-SAM and nnUNet2D after to_masks_dict().
    """
    def __init__(self):
        super().__init__()
        self.dc = MyDC()

    def forward(self, net_output, target):
        """
        net_output['masks']: (B, C, H, W) logits/probs
        target: (B, 1, H, W) with values in {0,1,2}
        """
        logits = net_output['masks']  # (B,C,H,W)

        # target one-hot: (B,H,W,C)
        target_oh = F.one_hot(target.squeeze(1).long(), 3).float()

        # pred one-hot: (B,H,W,C)
        pred = torch.argmax(logits, dim=1)          # (B,H,W)
        pred_oh = F.one_hot(pred.long(), 3).float() # (B,H,W,C)

        # flatten to (N,C)
        pred_oh = pred_oh.view(-1, 3)
        target_oh = target_oh.view(-1, 3)

        return self.dc(pred_oh, target_oh)


# =========================
# Main training function
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='nnUNet2D',
                        choices=['AoP-SAM', 'nnUNet2D'])
    parser.add_argument('--task', default='PSFH')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--encoder_input_size', type=int, default=256)
    parser.add_argument('--low_image_size', type=int, default=128)
    parser.add_argument('--enable_tri_encoder', action='store_true',
                    help='M1: input-level tri-branch shared encoder (default off)')
    parser.add_argument('--enable_weighted_ellipse', action='store_true',
                        help='M2: weighted ellipse fitting + confidence head (default off)')
    parser.add_argument('--enable_tta', action='store_true',
                        help='M3: test-time adaptation loop (default off)')
    parser.add_argument('--enable_ckpt_buffer_select', action='store_true',
                        help='M4: buffer snapshots + confidence-based selection (default off)')
    parser.add_argument('--enable_aug_aop_metric', action='store_true',
                        help='M5: aug_aop stability metric (default off)')

    # augmentation knobs (we will wire them in Step1; Step0 only defines them)
    parser.add_argument('--aug_brightness', type=float, default=0.0,
                        help='brightness jitter strength, 0 disables')
    parser.add_argument('--aug_contrast', type=float, default=0.0,
                        help='contrast jitter strength, 0 disables')
    parser.add_argument('--aug_gamma', type=float, default=0.0,
                        help='gamma jitter strength, 0 disables')
    parser.add_argument('--aug_noise_std', type=float, default=0.0,
                        help='gaussian noise std (0~1 after normalization), 0 disables')
    args = parser.parse_args()

    opt = get_config(args.task)
    device = torch.device(opt.device)

    # -------------------------
    # Reproducibility
    # -------------------------
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Transforms (不改)
    # -------------------------
    tf_train = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.5,
        p_rota=0.5,
        long_mask=True
    )

    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,
        long_mask=True
    )

    # -------------------------
    # Dataset & Loader (不改)
    # -------------------------
    train_set = ImageToImage2D(
        opt.data_path, 'train', tf_train,
        img_size=args.encoder_input_size
    )

    val_set = ImageToImage2D(
        opt.data_path, 'val', tf_val,
        img_size=args.encoder_input_size
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------------------------
    # Model
    # -------------------------
    args.enable_tri_encoder = False
    args.enable_weighted_ellipse = True
    args.enable_tta =True
    args.enable_ckpt_buffer_select = True
    args.enable_aug_aop_metric = True
    model = get_model(args=args).to(device)
    # -------------------------
    # Optimizer
    # NOTE: if nnUNet2D converges slow, consider weight_decay=1e-2 or 1e-3
    # -------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.base_lr,
        weight_decay=0.1
    )

    # -------------------------
    # Loss
    # -------------------------
    criterion = torch.nn.CrossEntropyLoss()

    # validation metric (unified): 1 - Dice
    dice_metric = DCloss()
    # optional extra for nnUNet2D val printing
    ce_metric = torch.nn.CrossEntropyLoss()

    os.makedirs('./checkpoints', exist_ok=True)
    best_score = 1e9  # lower is better (1 - dice)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            masks = map_mask_to_012(masks)
            out = model(imgs)                 # tensor or dict
            out = to_masks_dict(out)          # {'masks': logits}
            logits = out["masks"]             # (B,3,H,W)
            loss = criterion(logits, masks.squeeze(1).long())


            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(f"[Epoch {epoch+1}/{opt.epochs}] "
              f"Train loss: {total_loss/len(train_loader):.4f}")

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        val_dices = []
        val_ces = []

        img_idx = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                masks = batch['label'].to(device, non_blocking=True)   # GT: [B,1,H,W]

                if args.modelname == "nnUNet2D":
                    logits = model(imgs)

                    # 你原来的评估
                    out = to_masks_dict(logits)
                    val_dices.append(float(dice_metric(out, masks).item()))
                    if isinstance(logits, dict):
                        logits = logits["masks"]
                    val_ces.append(float(ce_metric(logits, masks.squeeze(1).long()).item()))

                    # ====== 新增：把预测变成“类别图/二值图” ======
                    pred = logits
                    if pred.ndim == 4 and pred.shape[1] > 1:
                        pred_mask = torch.argmax(pred, dim=1)          # [B,H,W]
                    elif pred.ndim == 4 and pred.shape[1] == 1:
                        pred_mask = (torch.sigmoid(pred) > 0.5).to(torch.uint8)  # [B,1,H,W]
                    else:
                        pred_mask = pred

                else:
                    out = model(imgs, None, None, None)
                    out = to_masks_dict(out)

                    # 你原来的评估
                    val_dices.append(float(dice_metric(out, masks).item()))

                    # ====== 新增：从 out 里拿 pred ======
                    pred = out["masks"] if isinstance(out, dict) and "masks" in out else out
                    if pred.ndim == 4 and pred.shape[1] > 1:
                        pred_mask = torch.argmax(pred, dim=1)          # [B,H,W]
                    elif pred.ndim == 4 and pred.shape[1] == 1:
                        pred_mask = (torch.sigmoid(pred) > 0.5).to(torch.uint8)  # [B,1,H,W]
                    else:
                        pred_mask = pred

        val_score = float(np.mean(val_dices))  # 1 - dice
        if len(val_ces) > 0:
            print(f"           Val (1-dice): {val_score:.4f} | Val CE: {float(np.mean(val_ces)):.4f}")
        else:
            print(f"           Val (1-dice): {val_score:.4f}")

        if val_score < best_score:
            best_score = val_score
            save_path = f'./checkpoints/{args.modelname}_true_final_no_1_{epoch}.pth'

            ckpt = {
                "state_dict": model.state_dict(),
                "cfg": vars(args),          # ✅ 保存所有开关/超参
            }
            torch.save(ckpt, save_path)
            print(f"           ✔ Saved best model to {save_path}")


if __name__ == '__main__':
    main()
