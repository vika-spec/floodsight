"""Flood change segmentation networks (pre + post Sentinel-1, plus terrain slope).

The production model is the Siamese U-Net. Two alternatives with the SAME encoder and decoder exist for the
model comparison (scripts/compare_models.py), so only the way the two dates are combined differs:
  * siamese       - shared encoder per date; decoder gets [post features, |post - pre| features]
  * early_fusion  - the two dates stacked as 6 input channels into one encoder (Daudt et al. 2018, FC-EF style)
  * post_only     - the after image only (single-image segmentation, like Sen1Floods11 baselines)
Further alternatives from segmentation-models-pytorch (pip install segmentation-models-pytorch), each made a
two-date change model the same way as ours (shared encoder, [post, |post - pre|] fused at every scale):
  * siamese_unetpp      - U-Net++ nested skip connections (Zhou et al. 2018; the idea behind SNUNet-CD)
  * siamese_attn_unet   - U-Net with scSE attention in the decoder (Roy et al. 2018)
  * siamese_deeplabv3p  - DeepLabV3+ with atrous spatial pyramid pooling (Chen et al. 2018)
  * siamese_segformer   - SegFormer MiT-B2 transformer encoder (Xie et al. 2021); with the temporal
                          difference this is the ChangeFormer idea (Bandara & Patel 2022)
  * segformer_early     - plain SegFormer MiT-B2 with the two dates stacked (single-network baseline)


Both images go through the SAME ResNet encoder (shared weights), so the network compares like with like.
At every scale the decoder receives [post features, |post - pre| features]: what the ground looks like now
and how much it changed. Output: 3 classes (0 no water, 1 permanent water, 2 flood).

Sized for an 8 GB laptop GPU: ResNet-34 encoder, 256 px tiles, batch 8, mixed precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder(name: str, in_ch: int, pretrained: bool):
    import torchvision

    ctor = getattr(torchvision.models, name)
    weights = "DEFAULT" if pretrained else None
    m = ctor(weights=weights)
    if in_ch != 3:
        w = m.conv1.weight.data
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        m.conv1.weight.data = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
    return m


def _channels(e) -> list[int]:
    """Feature channels at 1/2, 1/4, 1/8, 1/16, 1/32 for any torchvision ResNet (18/34: 64-512; 50/101: 64-2048)."""
    def out(layer):
        b = layer[-1]
        return (b.conv3 if hasattr(b, "conv3") else b.conv2).out_channels
    return [e.conv1.out_channels, out(e.layer1), out(e.layer2), out(e.layer3), out(e.layer4)]


def _decoder(self, chs, n_classes):
    self.up4 = ConvBlock(chs[4] + chs[3], 256)
    self.up3 = ConvBlock(256 + chs[2], 128)
    self.up2 = ConvBlock(128 + chs[1], 64)
    self.up1 = ConvBlock(64 + chs[0], 64)
    self.up0 = ConvBlock(64, 32)
    self.head = nn.Conv2d(32, n_classes, 1)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class SiameseUNet(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = 3, encoder: str = "resnet34", pretrained: bool = False):
        super().__init__()
        e = _encoder(encoder, in_ch, pretrained)
        self.stem = nn.Sequential(e.conv1, e.bn1, e.relu)           # 1/2, 64
        self.pool = e.maxpool
        self.l1, self.l2, self.l3, self.l4 = e.layer1, e.layer2, e.layer3, e.layer4  # 1/4..1/32
        chs = _channels(e)                                           # [64, 64, 128, 256, 512] for ResNet-34
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c),
                                                 nn.ReLU(inplace=True)) for c in chs])
        _decoder(self, chs, n_classes)

    def encode(self, x):
        f0 = self.stem(x)
        f1 = self.l1(self.pool(f0))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return [f0, f1, f2, f3, f4]

    def forward(self, pre, post):
        a, b = self.encode(pre), self.encode(post)
        f = [fu(torch.cat([bb, (bb - aa).abs()], 1)) for fu, aa, bb in zip(self.fuse, a, b)]

        def up(x, skip, block):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return block(torch.cat([x, skip], 1))

        x = up(f[4], f[3], self.up4)
        x = up(x, f[2], self.up3)
        x = up(x, f[1], self.up2)
        x = up(x, f[0], self.up1)
        x = F.interpolate(x, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up0(x))


class _SingleStreamUNet(nn.Module):
    """One encoder, same decoder as SiameseUNet (no per-date fusion)."""

    def __init__(self, in_ch: int, n_classes: int = 3, encoder: str = "resnet34", pretrained: bool = False):
        super().__init__()
        e = _encoder(encoder, in_ch, pretrained)
        self.stem = nn.Sequential(e.conv1, e.bn1, e.relu)
        self.pool = e.maxpool
        self.l1, self.l2, self.l3, self.l4 = e.layer1, e.layer2, e.layer3, e.layer4
        _decoder(self, _channels(e), n_classes)

    def _forward(self, x):
        f0 = self.stem(x)
        f1 = self.l1(self.pool(f0))
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)

        def up(a, skip, block):
            a = F.interpolate(a, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return block(torch.cat([a, skip], 1))

        y = up(f4, f3, self.up4)
        y = up(y, f2, self.up3)
        y = up(y, f1, self.up2)
        y = up(y, f0, self.up1)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up0(y))


class EarlyFusionUNet(_SingleStreamUNet):
    def __init__(self, in_ch: int = 3, n_classes: int = 3, encoder: str = "resnet34", pretrained: bool = False):
        super().__init__(2 * in_ch, n_classes, encoder, pretrained)

    def forward(self, pre, post):
        return self._forward(torch.cat([pre, post], 1))


class PostOnlyUNet(_SingleStreamUNet):
    def forward(self, pre, post):
        return self._forward(post)


class SmpChange(nn.Module):
    """Any segmentation-models-pytorch architecture as a two-date change model.

    mode 'siamese': one shared encoder per date; at every scale [post, |post - pre|] -> 1x1 conv back to the
    encoder's width (exactly the fusion of SiameseUNet), then the library's own decoder and head.
    mode 'early': the two dates stacked as 2*in_ch channels into a single network."""

    SPEC = {"siamese_unetpp": ("UnetPlusPlus", "resnet34", {}, "siamese"),
            "siamese_attn_unet": ("Unet", "resnet34", {"decoder_attention_type": "scse"}, "siamese"),
            "siamese_deeplabv3p": ("DeepLabV3Plus", "resnet34", {}, "siamese"),
            "siamese_segformer": ("Segformer", "mit_b2", {}, "siamese"),
            "segformer_early": ("Segformer", "mit_b2", {}, "early")}

    def __init__(self, arch: str, in_ch: int = 3, n_classes: int = 3, encoder: str | None = None,
                 pretrained: bool = False):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(f"architecture {arch!r} needs: pip install segmentation-models-pytorch") from e
        cls_name, enc, kw, self.mode = self.SPEC[arch]
        if encoder not in (None, "resnet34") and not enc.startswith("mit_"):
            enc = encoder                 # a different CNN encoder may be chosen; SegFormer keeps its MiT encoder
        m = getattr(smp, cls_name)(encoder_name=enc, encoder_weights="imagenet" if pretrained else None,
                                   in_channels=in_ch * (2 if self.mode == "early" else 1), classes=n_classes, **kw)
        self.encoder, self.decoder, self.head = m.encoder, m.decoder, m.segmentation_head
        if self.mode == "siamese":
            self.fuse = nn.ModuleList([
                nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)) if c
                else nn.Identity() for c in self.encoder.out_channels])

    def forward(self, pre, post):
        if self.mode == "early":
            f = self.encoder(torch.cat([pre, post], 1))
        else:
            a, b = self.encoder(pre), self.encoder(post)
            f = [bb if bb.shape[1] == 0 else fu(torch.cat([bb, (bb - aa).abs()], 1))
                 for fu, aa, bb in zip(self.fuse, a, b)]
        y = self.head(self.decoder(f))
        if y.shape[-2:] != post.shape[-2:]:
            y = F.interpolate(y, size=post.shape[-2:], mode="bilinear", align_corners=False)
        return y


ARCHS = {"siamese": SiameseUNet, "early_fusion": EarlyFusionUNet, "post_only": PostOnlyUNet,
         **{k: k for k in SmpChange.SPEC}}


def build_model(arch: str = "siamese", encoder: str = "resnet34", pretrained: bool = False, in_ch: int = 3,
                n_classes: int = 3):
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {', '.join(ARCHS)}")
    if arch in SmpChange.SPEC:
        return SmpChange(arch, in_ch, n_classes, encoder, pretrained)
    return ARCHS[arch](in_ch, n_classes, encoder, pretrained)


# --- input normalisation shared by training and inference --------------------------------------------

DB_MEAN = {"vv": -11.0, "vh": -18.0}
DB_STD = {"vv": 5.0, "vh": 5.0}


def normalise(vv_db, vh_db, slope_deg):
    """Stack one date into the 3-channel network input. Works on numpy arrays or tensors."""
    lib = torch if isinstance(vv_db, torch.Tensor) else __import__("numpy")
    vv = (lib.nan_to_num(vv_db, nan=DB_MEAN["vv"]) - DB_MEAN["vv"]) / DB_STD["vv"]
    vh = (lib.nan_to_num(vh_db, nan=DB_MEAN["vh"]) - DB_MEAN["vh"]) / DB_STD["vh"]
    sl = lib.nan_to_num(slope_deg, nan=0.0) / 30.0
    return lib.stack([vv, vh, sl])
