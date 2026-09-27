import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gn(ch):
    g = min(32, max(1, ch // 4))
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class ConvBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
            _gn(co),
            nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            _gn(co),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


def make_dct_filters():
    basis = []
    for u in range(8):
        for v in range(8):
            f = torch.zeros(8, 8)
            au = math.sqrt(1/8) if u == 0 else math.sqrt(2/8)
            av = math.sqrt(1/8) if v == 0 else math.sqrt(2/8)
            for x in range(8):
                for y in range(8):
                    f[x, y] = (
                        au * av
                        * math.cos((2*x+1)*u*math.pi/16)
                        * math.cos((2*y+1)*v*math.pi/16)
                    )
            basis.append(f)
    return torch.stack(basis)[:, None]


class LightweightJPEGEvidenceStream(nn.Module):
    """
    Independent JPEG/frequency stream.

    No CAT-Net modules.
    No CAT-Net pretrained weights.
    No HRNet.

    Input:
        image  : [B,3,H,W], RGB in [0,1]
        qtable : [B,1,8,8] or [B,8,8]

    Output:
        /4  : 64
        /8  : 128
        /16 : 256
        /32 : 512
    """

    channels = (64, 128, 256, 512)

    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.register_buffer("dct_filters", make_dct_filters())

        uv = [(u, v) for u in range(8) for v in range(8)]

        groups = {
            "low": [],
            "mid": [],
            "high": [],
            "horizontal": [],
            "vertical": [],
        }

        for i, (u, v) in enumerate(uv):
            if u == 0 and v == 0:
                continue

            s = u + v

            if s <= 2:
                groups["low"].append(i)
            elif s <= 5:
                groups["mid"].append(i)
            else:
                groups["high"].append(i)

            if u <= 1 and v >= 2:
                groups["horizontal"].append(i)

            if v <= 1 and u >= 2:
                groups["vertical"].append(i)

        self.groups = list(groups.values())

        # 5 local frequency maps + 3 qtable descriptors
        self.stem = ConvBlock(8, 96)

        self.p1 = nn.Conv2d(96, c1, 1)
        self.p2 = nn.Conv2d(96, c2, 1)

        self.stage3 = ConvBlock(96, 192, stride=2)
        self.p3 = nn.Conv2d(192, c3, 1)

        self.stage4 = ConvBlock(192, 256, stride=2)
        self.p4 = nn.Conv2d(256, c4, 1)

    def _frequency_maps(self, image):
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        y = 0.299*r + 0.587*g + 0.114*b

        h = (y.shape[-2] // 8) * 8
        w = (y.shape[-1] // 8) * 8
        y = y[..., :h, :w]

        coef = F.conv2d(
            y,
            self.dct_filters.to(dtype=y.dtype),
            stride=8
        )

        # Keep magnitude information; do NOT per-image standardize here.
        coef = torch.log1p(coef.abs())

        maps = [
            coef[:, ids].mean(1, keepdim=True)
            for ids in self.groups
        ]

        return torch.cat(maps, dim=1)

    @staticmethod
    def _qtable_maps(qtable, spatial_size, dtype, device):
        if qtable.ndim == 3:
            qtable = qtable.unsqueeze(1)

        q = qtable[:, :1].to(device=device, dtype=dtype)

        # log compression stabilizes scale while preserving JPEG quality.
        q = torch.log1p(q)

        qmean = q.mean((2, 3), keepdim=True)
        qstd = q.std((2, 3), keepdim=True, unbiased=False)
        qhigh = q[:, :, 4:, 4:].mean((2, 3), keepdim=True)

        desc = torch.cat([qmean, qstd, qhigh], dim=1)

        return desc.expand(
            -1, -1,
            spatial_size[0],
            spatial_size[1]
        )

    def forward(self, image: Tensor, qtable: Tensor):
        freq = self._frequency_maps(image)

        qmaps = self._qtable_maps(
            qtable,
            freq.shape[-2:],
            freq.dtype,
            freq.device
        )

        x = torch.cat([freq, qmaps], dim=1)

        x2 = self.stem(x)       # /8
        x3 = self.stage3(x2)    # /16
        x4 = self.stage4(x3)    # /32

        f1 = self.p1(
            F.interpolate(
                x2,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )                       # /4

        f2 = self.p2(x2)        # /8
        f3 = self.p3(x3)        # /16
        f4 = self.p4(x4)        # /32

        return [f1, f2, f3, f4]


class ReliabilityGatedForensicFusion(nn.Module):
    """
    Residual-preserving frequency injection.

    output = residual + gate * frequency_correction
    """

    def __init__(self, channels):
        super().__init__()

        self.freq_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.GELU()
        )

        self.correction = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                3,
                padding=1,
                bias=False
            ),
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

        # Start near residual-only behavior.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, residual, freq, available):
        if freq.shape[-2:] != residual.shape[-2:]:
            freq = F.interpolate(
                freq,
                size=residual.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        f = self.freq_proj(freq)

        x = torch.cat([residual, f], dim=1)

        correction = self.correction(x)

        gate = (
            self.gate(x)
            * available[:, :, None, None]
        )

        out = residual + gate * correction

        return out, residual, f
