from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_dct_basis() -> Tensor:
    """Orthonormal 8x8 DCT-II basis: [64,1,8,8]."""
    C = torch.zeros(8, 8, dtype=torch.float32)

    for u in range(8):
        alpha = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        for x in range(8):
            C[u, x] = alpha * math.cos(math.pi * (2 * x + 1) * u / 16.0)

    basis = []
    for u in range(8):
        for v in range(8):
            basis.append(torch.outer(C[u], C[v]))

    return torch.stack(basis, 0).unsqueeze(1)


class LocalFrequencyCue(nn.Module):
    """
    Lightweight, CAT-independent local frequency cue extractor.

    RGB
      -> grayscale
      -> fixed 8x8 DCT
      -> 8 grouped spectral maps
      -> lightweight convolutional embedding

    No CAT-Net code, HRNet, DCT-volume representation, qtable branch,
    or CAT-Net pretrained weights are used.
    """

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()

        self.register_buffer("dct_basis", _make_dct_basis(), persistent=False)

        coords = [(u, v) for u in range(8) for v in range(8)]

        def idx_where(fn):
            return torch.tensor(
                [i for i, (u, v) in enumerate(coords) if fn(u, v)],
                dtype=torch.long,
            )

        # AC groups.
        self.register_buffer(
            "idx_low",
            idx_where(lambda u, v: (u, v) != (0, 0) and (u + v) <= 3),
            persistent=False,
        )
        self.register_buffer(
            "idx_mid",
            idx_where(lambda u, v: 4 <= (u + v) <= 7),
            persistent=False,
        )
        self.register_buffer(
            "idx_high",
            idx_where(lambda u, v: (u + v) >= 8),
            persistent=False,
        )
        self.register_buffer(
            "idx_hori",
            idx_where(lambda u, v: (u, v) != (0, 0) and v > u),
            persistent=False,
        )
        self.register_buffer(
            "idx_vert",
            idx_where(lambda u, v: (u, v) != (0, 0) and u > v),
            persistent=False,
        )
        self.register_buffer(
            "idx_diag",
            idx_where(
                lambda u, v:
                (u, v) != (0, 0) and abs(u - v) <= 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "idx_ac",
            torch.arange(1, 64, dtype=torch.long),
            persistent=False,
        )

        g = 8 if embed_dim % 8 == 0 else 4

        self.encoder = nn.Sequential(
            nn.Conv2d(8, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, embed_dim),
            nn.GELU(),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                3,
                padding=1,
                groups=embed_dim,
                bias=False,
            ),
            nn.GroupNorm(g, embed_dim),
            nn.GELU(),

            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
        )

    @staticmethod
    def _energy(coef: Tensor, index: Tensor) -> Tensor:
        return coef[:, index].abs().mean(1, keepdim=True)

    def forward(self, image: Tensor) -> Tensor:
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Pad only enough to permit complete 8x8 blocks.
        h, w = gray.shape[-2:]
        ph = (-h) % 8
        pw = (-w) % 8
        if ph or pw:
            gray = F.pad(gray, (0, pw, 0, ph), mode="replicate")

        coef = F.conv2d(
            gray,
            self.dct_basis.to(dtype=gray.dtype),
            stride=8,
        )

        # Signed-log compression prevents very large coefficients
        # from dominating the spectral statistics.
        coef = torch.sign(coef) * torch.log1p(coef.abs())

        maps = torch.cat(
            [
                coef[:, 0:1].abs(),
                self._energy(coef, self.idx_low),
                self._energy(coef, self.idx_mid),
                self._energy(coef, self.idx_high),
                self._energy(coef, self.idx_hori),
                self._energy(coef, self.idx_vert),
                self._energy(coef, self.idx_diag),
                self._energy(coef, self.idx_ac),
            ],
            dim=1,
        )

        # Per-image/channel normalization.
        mean = maps.mean((2, 3), keepdim=True)
        var = (maps - mean).square().mean((2, 3), keepdim=True)
        maps = (maps - mean) * torch.rsqrt(var + 1e-6)

        return self.encoder(maps)


class FrequencyAwareQTR(nn.Module):
    """
    Wrap the existing QTR without changing its internal implementation.

    Local frequency cues are injected only into the three forensic
    feature scales consumed by QTR.
    """

    def __init__(
        self,
        base_qtr: nn.Module,
        detail_channels=(64, 128, 256),
        freq_dim: int = 32,
    ) -> None:
        super().__init__()

        self.base = base_qtr
        self.frequency = LocalFrequencyCue(freq_dim)

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(freq_dim, c, 1, bias=False)
                for c in detail_channels
            ]
        )

        # Small initial contribution; learned independently per scale.
        self.freq_alpha = nn.Parameter(torch.full((3,), -2.0))

    @property
    def use_reliability_gate(self):
        return self.base.use_reliability_gate

    @property
    def residual_refinement(self):
        return self.base.residual_refinement

    def forward(self, image, content, detail):
        fq = self.frequency(image)

        enhanced = list(detail)

        spatial_cues = []
        for i in range(3):
            x = F.interpolate(
                fq,
                size=detail[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = self.projections[i](x)

            alpha = torch.sigmoid(self.freq_alpha[i])
            enhanced[i] = detail[i] + alpha * x
            spatial_cues.append(x)

        out = self.base(image, content, enhanced)
        out["frequency_features"] = spatial_cues
        out["frequency_alpha"] = torch.sigmoid(self.freq_alpha)

        return out
