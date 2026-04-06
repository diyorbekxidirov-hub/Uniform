"""
models/model.py  —  EfficientNet-B0 + CBAM + GeM + Embedding Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from configs.config import EMBED_DIM, PRETRAINED, USE_CBAM


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * att.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.pool   = GeM()
        self.bn     = nn.BatchNorm1d(in_features)
        self.drop   = nn.Dropout(p=0.3)
        self.fc     = nn.Linear(in_features, embed_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn_out(x)
        return F.normalize(x, p=2, dim=1)


class UniformEmbedNet(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM,
                 pretrained: bool = PRETRAINED,
                 use_cbam: bool = USE_CBAM,
                 freeze: bool = False):
        super().__init__()
        weights       = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff           = efficientnet_b0(weights=weights)
        self.backbone = eff.features        # [B, 1280, H, W]
        in_ch         = 1280
        self.cbam     = CBAM(in_ch) if use_cbam else nn.Identity()
        self.head     = EmbeddingHead(in_ch, embed_dim)
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[Model] Backbone FROZEN")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[Model] Backbone UNFROZEN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.cbam(x)
        return self.head(x)


def build_model(device: str = "cuda", freeze: bool = False) -> UniformEmbedNet:
    model   = UniformEmbedNet(freeze=freeze).to(device)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] EfficientNet-B0 + CBAM + GeM  "
          f"embed={EMBED_DIM}  total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
    return model
