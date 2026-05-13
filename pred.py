import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import SimpleITK as sitk
from tqdm import tqdm

from models.model_dict import get_model
from utils.data_us import JointTransform2D, correct_dims
from utils.aop_confidence import aop_confidence_from_pred
import copy
import torch.nn as nn
import torch.nn.functional as F

def unwrap_logits(out):
    # AoP-SAM -> dict['masks']; nnUNet2D -> Tensor
    return out["masks"] if isinstance(out, dict) else out

def disable_dropout_(model: nn.Module):
    # safety: keep dropout off even in train() during TTA
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()

def set_tta_trainable_(model: nn.Module):
    """
    Freeze everything, unfreeze ONLY:
      - Norm layers affine params (BN / IN / GN / LN if exists)
      - Linear params
    Returns the list of trainable parameters.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable = []

    NORM_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.GroupNorm, nn.LayerNorm,
    )

    for m in model.modules():
        #print(m)
        if isinstance(m, NORM_TYPES):
            # many norms have affine (weight/bias); some IN may have affine=False
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.requires_grad = True
                trainable.append(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
                trainable.append(m.bias)

        elif isinstance(m, nn.Linear):
            if m.weight is not None:
                m.weight.requires_grad = True
                trainable.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad = True
                trainable.append(m.bias)

    return trainable


def entropy_min_loss(logits: torch.Tensor):
    """
    Placeholder unsupervised loss: mean pixel-wise entropy of softmax
    logits: (B,C,H,W)
    """
    p = torch.softmax(logits, dim=1).clamp_min(1e-8)
    ent = -(p * torch.log(p)).sum(dim=1)  # (B,H,W)
    return ent.mean()

def tv_smoothness_loss(probs: torch.Tensor):
    """
    Optional smoothness on probabilities, probs: (B,C,H,W)
    """
    dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    return dh + dw

class TTAManager:
    def __init__(self, model: nn.Module, lr: float, restore_every: int, restore_prob: float,
                 entropy_w: float, tv_w: float):
        self.model = model
        self.lr = lr
        self.restore_every = max(int(restore_every), 1)
        self.restore_prob = float(restore_prob)
        self.entropy_w = float(entropy_w)
        self.tv_w = float(tv_w)

        # choose trainable params
        self.trainable = set_tta_trainable_(self.model)
        # snapshot pretrained weights for trainable tensors only (CPU)
        self.pretrained = {id(p): p.detach().cpu().clone() for p in self.trainable}

        self.opt = torch.optim.Adam(self.trainable, lr=self.lr)
        self.step_count = 0

    @torch.no_grad()
    def _partial_restore_(self):
        # coarse restore: restore whole tensors with prob p
        if self.restore_prob <= 0:
            return
        for p in self.trainable:
            if torch.rand((), device=p.device).item() < self.restore_prob:
                p.copy_(self.pretrained[id(p)].to(device=p.device, dtype=p.dtype))

    def adapt(self, imgs: torch.Tensor, forward_fn):
        """
        forward_fn(imgs) -> out (dict or tensor), we unwrap to logits inside.
        """
        self.model.train()
        disable_dropout_(self.model)

        for _ in range(self._steps_per_batch):
            out = forward_fn(imgs)
            logits = unwrap_logits(out)  # (B,C,H,W)

            loss = self.entropy_w * entropy_min_loss(logits)

            if self.tv_w > 0:
                probs = torch.softmax(logits, dim=1)
                loss = loss + self.tv_w * tv_smoothness_loss(probs)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            self.step_count += 1
            if (self.step_count % self.restore_every) == 0:
                self._partial_restore_()

        self.model.eval()

    @property
    def _steps_per_batch(self):
        # will be set from args later (avoid passing args everywhere)
        return getattr(self, "steps_per_batch", 1)

def to_masks_dict(out):
    #raise ValueError(out)
    # AoP-SAM -> dict
    if isinstance(out, dict) and "masks" in out:
        return out
    # nnUNet2D -> tensor (B,C,H,W)
    return {"masks": out}


def ensure_hwc(arr: np.ndarray) -> np.ndarray:
    """
    兼容 SITK 读出来的各种形状，统一成 (H, W, C)
    """
    if arr.ndim == 2:
        return arr[:, :, None]  # (H,W,1)

    if arr.ndim == 3:
        # 常见：SITK -> (C,H,W) 或 (1,H,W)
        if arr.shape[0] in (1, 3) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return np.transpose(arr, (1, 2, 0))  # (H,W,C)
        # 也可能已经是 (H,W,C)
        if arr.shape[-1] in (1, 3) and arr.shape[0] > 8 and arr.shape[1] > 8:
            return arr
    # 兜底：不做强假设
    return arr


class FetalDatasetLazy(Dataset):
    """
    Lazy-loading：
    - init 不读 mha
    - __getitem__ 每次只读一个样本
    """

    def __init__(self, image_files, label_files, transform=None, target_size=(256, 256)):
        assert len(image_files) == len(label_files), "image_files and label_files must have the same length"
        self.image_files = list(image_files)
        self.label_files = list(label_files)
        self.transform = transform
        self.target_size = tuple(target_size)

        print("-" * 20)
        print(f"FetalDatasetLazy: {len(self.image_files)} samples")
        print(f"target_size: {self.target_size}")
        print("Lazy loading enabled")
        print("-" * 20)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # -------- read mha on demand --------
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(str(self.image_files[idx])))
        mask_np  = sitk.GetArrayFromImage(sitk.ReadImage(str(self.label_files[idx])))

        # -------- shape unify: (H,W,C) --------
        image_np = ensure_hwc(image_np)
        mask_np  = ensure_hwc(mask_np)

        # correct_dims 会把 (H,W) -> (H,W,1)
        image_np, mask_np = correct_dims(image_np, mask_np)

        # -------- resize --------
        image_np = cv2.resize(image_np, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask_np  = cv2.resize(mask_np,  self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        # 注意：mask_np 可能是 (H,W,1)，这里 squeeze 后映射再 unsqueeze 回来
        m = mask_np.squeeze(-1) if (mask_np.ndim == 3 and mask_np.shape[-1] == 1) else mask_np
        if m.ndim == 2:
            mask_np = m[:, :, None]
        else:
            mask_np = m

        # 如果 image 是 1 通道，复制成 3 通道（与你 in_channels=3 对齐）
        if image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        # -------- transform --------
        if self.transform is not None:
            image_t, mask_t, low_mask_t = self.transform(image_np, mask_np)
        else:
            # 理论上你会传 JointTransform2D，这里只是兜底
            image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_np.squeeze(-1)).long()
            low_mask_t = None

        sample = {
            "image": image_t,                 # (C,H,W) float
            "label": mask_t.unsqueeze(0),     # (1,H,W) long, values 0/1/2
        }
        if low_mask_t is not None:
            sample["low_res_label"] = low_mask_t.unsqueeze(0)

        return sample

class SnapshotBuffer:
    """
    Keep rolling snapshots of model weights on CPU.
    Save every k steps, keep at most c snapshots.
    """
    def __init__(self, model: nn.Module, k: int, c: int):
        self.model = model
        self.k = max(int(k), 1)
        self.c = max(int(c), 1)
        self.snaps = []  # list of dicts: {"step": int, "state": state_dict}

    @torch.no_grad()
    def _clone_state_cpu(self):
        sd = self.model.state_dict()
        out = {}
        for key, val in sd.items():
            if torch.is_tensor(val):
                out[key] = val.detach().cpu().clone()
            else:
                out[key] = val
        return out

    @torch.no_grad()
    def maybe_save(self, step: int):
        if step <= 0:
            return
        if (step % self.k) != 0:
            return
        self.snaps.append({"step": int(step), "state": self._clone_state_cpu()})
        if len(self.snaps) > self.c:
            self.snaps = self.snaps[-self.c:]

    def __len__(self):
        return len(self.snaps)

    @torch.no_grad()
    def load(self, state_cpu: dict):
        # load CPU tensors into current GPU model params
        self.model.load_state_dict(state_cpu, strict=False)

    def iter_states(self):
        # yield older -> newer (or you can reverse if you prefer)
        for item in self.snaps:
            yield item["step"], item["state"]

def apply_intensity_perturb(imgs: torch.Tensor, args, seed: int):
    """
    imgs: (B,C,H,W) float. usually in [0,1] after transform.
    Apply small non-spatial perturbations: brightness/contrast/gamma/noise.
    """
    x = imgs
    # detect range
    x_max = float(x.detach().max().item())
    lo, hi = (0.0, 1.0) if x_max <= 1.5 else (0.0, 255.0)

    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed))

    # brightness: add delta in [-b, b] * (hi-lo)
    b = float(getattr(args, "aug_brightness", 0.0))
    if b > 0:
        delta = (torch.rand((x.shape[0], 1, 1, 1), generator=g, device=x.device) * 2 - 1.0) * b * (hi - lo)
        x = x + delta

    # contrast: scale around per-image mean, factor in [1-c, 1+c]
    c = float(getattr(args, "aug_contrast", 0.0))
    if c > 0:
        factor = 1.0 + (torch.rand((x.shape[0], 1, 1, 1), generator=g, device=x.device) * 2 - 1.0) * c
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * factor + mean

    # gamma: power transform on normalized range
    gm = float(getattr(args, "aug_gamma", 0.0))
    if gm > 0:
        gamma = 1.0 + (torch.rand((x.shape[0], 1, 1, 1), generator=g, device=x.device) * 2 - 1.0) * gm
        # normalize -> apply gamma -> de-normalize
        xn = (x - lo) / max(hi - lo, 1e-6)
        xn = xn.clamp(0.0, 1.0)
        xn = xn.pow(gamma)
        x = xn * (hi - lo) + lo

    # noise: gaussian
    ns = float(getattr(args, "aug_noise_std", 0.0))
    if ns > 0:
        noise = torch.randn_like(x, generator=g) * (ns * (hi - lo))
        x = x + noise

    x = x.clamp(lo, hi)
    return x
def refine_pred_cc_with_conf(pred_012: np.ndarray,
                             conf_2hw: np.ndarray,
                             min_area: int = 30,
                             min_conf: float = 0.0,
                             keep_largest: bool = False) -> np.ndarray:
    """
    pred_012: (H,W) in {0,1,2}
    conf_2hw: (2,H,W) confidence map for class1/class2, or None
    """
    out = pred_012.copy().astype(np.uint8)

    # we process class1 then class2, so class2 will override if overlap happens
    for cls in (1, 2):
        binmask = (out == cls).astype(np.uint8)
        if binmask.sum() == 0:
            continue

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binmask, connectivity=8)
        if num <= 1:
            continue

        kept = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue

            if conf_2hw is not None:
                conf_map = conf_2hw[cls - 1]  # class1->0, class2->1
                mconf = float(conf_map[labels == i].mean())
                if mconf < float(min_conf):
                    continue
            else:
                mconf = 1.0

            kept.append((i, mconf, area))

        # remove all components of this class first
        out[binmask == 1] = 0

        if len(kept) == 0:
            continue

        if keep_largest:
            # choose by conf first, then area
            best_i = max(kept, key=lambda x: (x[1], x[2]))[0]
            out[labels == best_i] = cls
        else:
            for i, _, _ in kept:
                out[labels == i] = cls

    return out

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", default="nnUNet2D", choices=["nnUNet2D"])
    parser.add_argument("--ckpt", type=str, help="trained checkpoint path (.pth)")
    parser.add_argument("--task", default="PSFH")
    parser.add_argument("--encoder_input_size", type=int, default=256)
    parser.add_argument("--low_image_size", type=int, default=128)

    # AoP-SAM init uses this in get_model(args) sometimes
    parser.add_argument("--vit_name", type=str, default="vit_h")
    parser.add_argument("--sam_ckpt", type=str, default="", help="SAM pretrained ckpt for AoP-SAM init (optional)")

    # data
    parser.add_argument("--data_root", type=str, default="/FM_data/cyf/shared_data/test_2025")
    parser.add_argument("--out_root", type=str, default="./out")
    parser.add_argument("--fix", type=str, default="no_2_2025_new")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
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


    parser.add_argument('--tta_steps', type=int, default=1)
    parser.add_argument('--tta_lr', type=float, default=1e-4)
    parser.add_argument('--tta_restore_every', type=int, default=20)
    parser.add_argument('--tta_restore_prob', type=float, default=0.02)
    parser.add_argument('--tta_entropy_weight', type=float, default=1.0)
    parser.add_argument('--tta_tv_weight', type=float, default=0.0)


    parser.add_argument('--ckpt_buffer_k', type=int, default=20,
                    help='M4: save snapshot every k TTA steps')
    parser.add_argument('--ckpt_buffer_c', type=int, default=4,
                    help='M4: keep at most c snapshots in RAM')

    parser.add_argument('--aug_aop_n', type=int, default=4,
                    help='M5: number of small intensity perturbations per sample')

        # M6: post refine (connected components + optional confidence)
    parser.add_argument('--enable_refine_cc', action='store_true',
                        help='M6: confidence-guided connected-component refinement (default off)')
    parser.add_argument('--refine_min_area', type=int, default=30,
                        help='M6: remove components smaller than this area (pixels)')
    parser.add_argument('--refine_min_conf', type=float, default=0.0,
                        help='M6: remove components with mean conf < this (requires conf head)')
    parser.add_argument('--refine_keep_largest', action='store_true',
                        help='M6: keep only the best component per class (by conf then area)')

    return parser

def morph_open(binmask: np.ndarray, k: int = 3, it: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(binmask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=it)
    return out

def main():
    args = build_argparser().parse_args()
    args.ckpt = "/home/cyf/LLM/step1 copy/checkpoints/nnUNet2D_best_1.pth"
    # device
    args.device = "cuda:7"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # seeds
    seed_value = 1
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    # output dirs
    gt_dir = Path(args.out_root) / "gt" / args.fix
    pr_dir = Path(args.out_root) / "pred" / args.fix
    gt_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)

    # transforms (val)
    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=512,
        crop=None,
        p_flip=0.0,
        color_jitter_params=None,
        long_mask=True
    )

    # data list
    root_path = Path(args.data_root)

    img_dir = root_path / "image_mha"
    lab_dir = root_path / "label_mha"

    image_files = sorted(img_dir.glob("*.mha"))

    # mask 与 image 同名
    label_files = [lab_dir / f.name for f in image_files]

    # 可选：检查缺失标签
    missing = [str(p) for p in label_files if not p.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing {len(missing)} label files, e.g. {missing[:5]}")

    print(f"Found {len(image_files)} images and {len(label_files)} masks")
    print(f"Example pair:\n  img: {image_files[0]}\n  lab: {label_files[0]}")

    ds = FetalDatasetLazy(image_files=image_files, label_files=label_files, transform=tf_val, target_size=(256, 256))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # model
    # 注意：get_model 会用 args.modelname 选择模型
    # -------------------------
    # Load checkpoint (supports old/new format)
    # -------------------------
    raw = torch.load(args.ckpt, map_location="cpu")

    # 你可以加一个命令行开关控制是否忽略 cfg（可选）
    IGNORE_CKPT_CFG = getattr(args, "ignore_ckpt_cfg", False)

    cfg = {}
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        cfg = raw.get("cfg", {}) or {}
    else:
        state_dict = raw

    # ✅ 同步“所有 args 里存在的字段”，而不是只同步几个 enable_*
    if (not IGNORE_CKPT_CFG) and isinstance(cfg, dict) and len(cfg) > 0:
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if not (isinstance(cfg, dict) and "enable_tri_encoder" in cfg):
        args.enable_tri_encoder = False
    if not (isinstance(cfg, dict) and "enable_weighted_ellipse" in cfg):
        args.enable_weighted_ellipse = False
    #raise ValueError(args)
    # build model with synchronized args
    model = get_model(args=args).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded.")
    if len(missing) > 0:
        print("  [Missing keys]   ", missing[:10], "... total", len(missing))
    if len(unexpected) > 0:
        print("  [Unexpected keys]", unexpected[:10], "... total", len(unexpected))
    # ---- M3: init TTA manager (optional) ----
    tta_mgr = None
    if getattr(args, "enable_tta", False):
        tta_mgr = TTAManager(
            model=model,
            lr=args.tta_lr,
            restore_every=args.tta_restore_every,
            restore_prob=args.tta_restore_prob,
            entropy_w=args.tta_entropy_weight,
            tv_w=args.tta_tv_weight
        )
        tta_mgr.steps_per_batch = args.tta_steps
        print(f"[TTA] enabled: steps={args.tta_steps}, lr={args.tta_lr}, "
            f"restore_every={args.tta_restore_every}, restore_prob={args.tta_restore_prob}")

    snap_buf = None
    if getattr(args, "enable_ckpt_buffer_select", False):
        snap_buf = SnapshotBuffer(model=model, k=args.ckpt_buffer_k, c=args.ckpt_buffer_c)
        print(f"[M4] enabled: ckpt_buffer_select, k={args.ckpt_buffer_k}, c={args.ckpt_buffer_c}")

    model.eval()

    # =========================
    # inference (M4 + M5 + M6)
    # =========================

    # ---- M5 logs ----
    m5_std_all = []   # per-sample std of AoP over augmentations
    m5_rng_all = []   # per-sample range of AoP over augmentations

    # ---------- helper forward ----------
    def _forward(x):
        if args.modelname == "nnUNet2D":
            return model(x)
        else:
            return model(x, None, None, None)

    # ---------- helper: run M4 selection and return best_pred + best_aop + best_c_aop + best_conf ----------
    @torch.no_grad()
    def predict_best_with_m4(x, save_snapshot: bool):
        """
        x: (B,C,H,W) torch float
        Returns:
          best_pred_np: (B,H,W) uint8 in {0,1,2}
          best_aop:     (B,) float32  (NaN if conf not available)
          best_caop:    (B,) float32  (0 if conf not available)
          best_conf:    (B,2,H,W) float32 or None
        """
        # M4: snapshot saving (only once per batch, after TTA updates)
        if save_snapshot and (snap_buf is not None) and (tta_mgr is not None):
            snap_buf.maybe_save(tta_mgr.step_count)

        out_cur = to_masks_dict(_forward(x))
        logits_cur = out_cur["masks"]                    # (B,3,H,W)
        pred_cur_t = torch.argmax(logits_cur, dim=1)     # (B,H,W)
        conf_cur = out_cur.get("conf", None)             # (B,2,H,W) or None

        B = pred_cur_t.shape[0]
        best_pred_np = pred_cur_t.detach().cpu().numpy().astype(np.uint8)

        best_conf = None
        if conf_cur is not None:
            best_conf = conf_cur.detach().cpu().numpy().astype(np.float32)  # (B,2,H,W)

        best_caop = np.zeros((B,), dtype=np.float32)
        best_aop  = np.full((B,), np.nan, dtype=np.float32)
        best_score = np.full((B,), -1e18, dtype=np.float32)

        if conf_cur is not None:
            conf_np = conf_cur.detach().cpu().numpy()  # (B,2,H,W)
            for b in range(B):
                info = aop_confidence_from_pred(best_pred_np[b], conf_np[b])
                best_caop[b] = float(info["c_aop"])
                best_aop[b]  = float(info["aop"])
                best_score[b] = best_caop[b]
        else:
            best_score[:] = 0.0

        # evaluate snapshots if enabled and buffer not empty
        if (snap_buf is not None) and (len(snap_buf) > 0):
            cur_state = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items() if torch.is_tensor(v)}

            for step_id, st_cpu in snap_buf.iter_states():
                snap_buf.load(st_cpu)

                out_s = to_masks_dict(_forward(x))
                logits_s = out_s["masks"]
                pred_s_np = torch.argmax(logits_s, dim=1).detach().cpu().numpy().astype(np.uint8)
                conf_s = out_s.get("conf", None)
                if conf_s is None:
                    continue

                conf_np_s = conf_s.detach().cpu().numpy()  # (B,2,H,W)
                for b in range(B):
                    info = aop_confidence_from_pred(pred_s_np[b], conf_np_s[b])
                    s = float(info["c_aop"])
                    if s > best_score[b]:
                        best_score[b] = s
                        best_pred_np[b] = pred_s_np[b]
                        best_caop[b] = s
                        best_aop[b]  = float(info["aop"])
                        if best_conf is not None:
                            best_conf[b] = conf_np_s[b].astype(np.float32)

            model.load_state_dict(cur_state, strict=False)

        return best_pred_np, best_aop, best_caop, best_conf

    global_idx = 1
    for datapack in tqdm(dl, total=len(dl)):
        imgs = datapack["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = datapack["label"].to(device=device, dtype=torch.long, non_blocking=True)  # (B,1,H,W)
        #raise ValueError(tta_mgr)
        # ---- M3: TTA (only on original imgs) ----
        if tta_mgr is not None:
            def _fw_train(x):
                return _forward(x)
            tta_mgr.adapt(imgs, _fw_train)

        # ---- prediction: with M4 or without M4 ----
        with torch.no_grad():
            if snap_buf is not None:
                best_pred, best_aop, best_caop, best_conf = predict_best_with_m4(imgs, save_snapshot=True)
            else:
                out_cur = to_masks_dict(_forward(imgs))
                logits_cur = out_cur["masks"]
                #raise ValueError(logits_cur[0])
                best_pred = torch.argmax(logits_cur, dim=1).detach().cpu().numpy().astype(np.uint8)
                conf_cur = out_cur.get("conf", None)
                best_conf = None if conf_cur is None else conf_cur.detach().cpu().numpy().astype(np.float32)
                # M5需要aop的话，这里只能在有conf时计算
                B0 = best_pred.shape[0]
                best_aop  = np.full((B0,), np.nan, dtype=np.float32)
                best_caop = np.zeros((B0,), dtype=np.float32)
                if best_conf is not None:
                    for b in range(B0):
                        info = aop_confidence_from_pred(best_pred[b], best_conf[b])
                        best_aop[b]  = float(info["aop"])
                        best_caop[b] = float(info["c_aop"])

        B = best_pred.shape[0]

        # ---- M5: aug_aop stability metric ----
        if getattr(args, "enable_aug_aop_metric", False):
            if np.isfinite(best_aop).any():
                aop_runs = [best_aop.copy()]
                n_aug = int(getattr(args, "aug_aop_n", 4))
                for t in range(n_aug):
                    imgs_aug = apply_intensity_perturb(imgs, args, seed=int(global_idx * 1000 + t + 7))
                    with torch.no_grad():
                        if snap_buf is not None:
                            _, aop_aug, _, _ = predict_best_with_m4(imgs_aug, save_snapshot=False)
                        else:
                            out_aug = to_masks_dict(_forward(imgs_aug))
                            logits_aug = out_aug["masks"]
                            pred_aug = torch.argmax(logits_aug, dim=1).detach().cpu().numpy().astype(np.uint8)
                            conf_aug = out_aug.get("conf", None)
                            if conf_aug is None:
                                aop_aug = np.full((B,), np.nan, dtype=np.float32)
                            else:
                                conf_aug_np = conf_aug.detach().cpu().numpy().astype(np.float32)
                                aop_aug = np.full((B,), np.nan, dtype=np.float32)
                                for b in range(B):
                                    info = aop_confidence_from_pred(pred_aug[b], conf_aug_np[b])
                                    aop_aug[b] = float(info["aop"])
                        aop_runs.append(aop_aug.copy())

                aop_stack = np.stack(aop_runs, axis=0)  # (1+N, B)
                aop_std = np.nanstd(aop_stack, axis=0).astype(np.float32)
                aop_rng = (np.nanmax(aop_stack, axis=0) - np.nanmin(aop_stack, axis=0)).astype(np.float32)
                m5_std_all.append(aop_std)
                m5_rng_all.append(aop_rng)

        # ---- M6: post refine ----
        if getattr(args, "enable_refine_cc", False):
            for b in range(B):
                conf_b = None if best_conf is None else best_conf[b]  # (2,H,W) or None
                best_pred[b] = refine_pred_cc_with_conf(
                    best_pred[b],
                    conf_b,
                    min_area=args.refine_min_area,
                    min_conf=args.refine_min_conf,
                    keep_largest=getattr(args, "refine_keep_largest", False)
                )

        # ---- save pngs ----
        lab = masks.squeeze(1).detach().cpu().numpy().astype(np.uint8)  # (B,H,W)
        #best_pred = morph_open(best_pred, 3)
        for b in range(B):
            cv2.imwrite(str(pr_dir / f"{global_idx}.png"), best_pred[b] * 127)
            cv2.imwrite(str(gt_dir / f"{global_idx}.png"), lab[b] * 127)
            global_idx += 1

    # ---- M5 summary ----
    if getattr(args, "enable_aug_aop_metric", False) and len(m5_std_all) > 0:
        std_all = np.concatenate(m5_std_all, axis=0)
        rng_all = np.concatenate(m5_rng_all, axis=0)
        print(f"[M5] aug_aop_metric enabled, N_total={std_all.shape[0]}")
        print(f"[M5] AoP std : mean={float(np.nanmean(std_all)):.6f}, median={float(np.nanmedian(std_all)):.6f}")
        print(f"[M5] AoP range: mean={float(np.nanmean(rng_all)):.6f}, median={float(np.nanmedian(rng_all)):.6f}")

    print(f"Done. Saved pred to: {pr_dir}")
    print(f"Done. Saved gt   to: {gt_dir}")




if __name__ == "__main__":
    main()
