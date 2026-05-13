import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TriFusion(nn.Module):
    """Fuse three same-scale feature maps by concat + 1x1 compress."""
    def __init__(self, ch: int):
        super().__init__()
        self.compress = nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        return self.compress(torch.cat([f1, f2, f3], dim=1))


class nnUNet2D(nn.Module):
    """
    Baseline: simple 2D U-Net.
    M1 (optional): input-level tri-branch shared encoder.
    M2 (optional): confidence head for weighted ellipse fitting (two channels: pubic/head).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        enable_tri_encoder: bool = False,
        tri_mask_temp: float = 1.0,
        enable_weighted_ellipse: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.enable_tri_encoder = enable_tri_encoder
        self.tri_mask_temp = tri_mask_temp
        self.enable_weighted_ellipse = enable_weighted_ellipse

        # -------------------------
        # Shared Encoder
        # -------------------------
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # -------------------------
        # Decoder
        # -------------------------
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)
        #self.edge_head = nn.Conv2d(self.out_conv.in_channels, 1, kernel_size=1, bias=True)
        # -------------------------
        # M1: proposal head + tri-branch fusion
        # -------------------------
        if self.enable_tri_encoder:
            self.proposal_head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(base_channels, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(base_channels, 2, 1, bias=True),  # (B,2,H,W)
            )
            self.fuse1 = TriFusion(base_channels)
            self.fuse2 = TriFusion(base_channels * 2)
            self.fuse3 = TriFusion(base_channels * 4)
            self.fuse4 = TriFusion(base_channels * 8)

        # -------------------------
        # M2: confidence head (pubic/head), predicts weight map in (0,1)
        # Uses high-res feature (enc1 output) for stable boundaries.
        # -------------------------
        if self.enable_weighted_ellipse:
            self.conf_head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(base_channels, affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(base_channels, 2, 1, bias=True),  # (B,2,H,W)
            )

    # -------------------------
    # helpers
    # -------------------------
    def _encode_once(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def _decode(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor, b: torch.Tensor):
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out, d1

    @staticmethod
    def _soft_mask_from_logits(mask_logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        if temp is None or temp <= 0:
            temp = 1.0
        m = torch.sigmoid(mask_logits / temp)
        return m.clamp(0.0, 1.0)

    def _conf_from_e1(self, e1_full: torch.Tensor) -> torch.Tensor:
        # return sigmoid(conf_logits) in (0,1)
        conf_logits = self.conf_head(e1_full)  # (B,2,H,W)
        return torch.sigmoid(conf_logits).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor):
        """
        Returns:
          - if enable_weighted_ellipse=False: Tensor logits (B,C,H,W)  (baseline-compatible)
          - if enable_weighted_ellipse=True : dict {'masks': logits, 'conf': conf} where conf is (B,2,H,W)
        """
        if not self.enable_tri_encoder:
            e1, e2, e3, e4, b = self._encode_once(x)
            logits, d1 = self._decode(e1, e2, e3, e4, b)
            #edge = self.edge_head(d1)
            if not self.enable_weighted_ellipse:
                return logits
            conf = self._conf_from_e1(e1)
            return {"masks": logits, "conf": conf}
            #return {"masks": logits, "edge": edge}

        # tri-branch
        e1_full, e2_full, e3_full, e4_full, b_full = self._encode_once(x)

        mask_logits = self.proposal_head(e1_full)  # (B,2,H,W)
        soft = self._soft_mask_from_logits(mask_logits, self.tri_mask_temp)
        mh = soft[:, 0:1]
        mp = soft[:, 1:2]

        x_head  = x * mh
        x_pubic = x * mp

        e1_h, e2_h, e3_h, e4_h, b_h = self._encode_once(x_head)
        e1_p, e2_p, e3_p, e4_p, b_p = self._encode_once(x_pubic)

        e1 = self.fuse1(e1_full, e1_h, e1_p)
        e2 = self.fuse2(e2_full, e2_h, e2_p)
        e3 = self.fuse3(e3_full, e3_h, e3_p)
        e4 = self.fuse4(e4_full, e4_h, e4_p)

        b = (b_full + b_h + b_p) / 3.0
        logits, d1 = self._decode(e1, e2, e3, e4, b)

        if not self.enable_weighted_ellipse:
            return logits
            

        
        #edge = self.edge_head(logits)  # (B,1,H,W) logits

        conf = self._conf_from_e1(e1_full)  # use full-image e1 for conf stability
        return {"masks": logits, "conf": conf}

    def get_tri_masks(self, x: torch.Tensor):
        if not self.enable_tri_encoder:
            raise RuntimeError("Tri-encoder is disabled.")
        e1_full = self.enc1(x)
        mask_logits = self.proposal_head(e1_full)
        soft = self._soft_mask_from_logits(mask_logits, self.tri_mask_temp)
        return soft[:, 0:1], soft[:, 1:2]
