import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table, ActivationCountAnalysis
from models.nnunet_2d.nnunet_2d import nnUNet2D

def get_model(args):
    return nnUNet2D(
        in_channels=3,
        num_classes=3,
        base_channels=32,
        enable_tri_encoder=getattr(args, "enable_tri_encoder", True),
        enable_weighted_ellipse=getattr(args, "enable_weighted_ellipse", True),
    )




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Networks')

    parser.add_argument('-encoder_input_size', type=int, default=512)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--vit_name', type=str, default='vit_h')
    parser.add_argument('--sam_ckpt', type=str,
                        default="../checkpoints/sam_vit_h_4b8939.pth",
                        help='Pretrained checkpoint of SAM')

    args = parser.parse_args()
    model = get_model(args=args)

    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, ActivationCountAnalysis

    x = torch.randn(1, 3, 512, 512)
    flops = FlopCountAnalysis(model, x)
    print(flops.total())
    print(parameter_count_table(model))
    outPack = model(x)
    print(outPack['low_res_logits'].shape)
    print(outPack['masks'].shape)
