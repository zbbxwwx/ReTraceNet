"""Native-JPEG forensic HRNet used by the two-stream ReTraceNet.

The module is independent of CAT-Net code and weights.  It consumes the decoded
RGB image, the native quantized Y-channel 8x8 coefficient grid, and its qtable.
Residual and JPEG evidence meet once at the high-resolution forensic branch;
all later processing happens inside one HRNet-style encoder.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import ConstrainedResidualBank


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = ConvNormAct(channels, channels, activation=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class DownsamplePath(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, steps: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for index in range(steps):
            target = out_channels if index == steps - 1 else current
            layers.append(ConvNormAct(current, target, stride=2))
            current = target
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiResolutionExchange(nn.Module):
    """One HRNet exchange unit with persistent parallel resolutions."""

    def __init__(self, channels: Sequence[int], blocks_per_branch: int = 2) -> None:
        super().__init__()
        self.channels = tuple(int(value) for value in channels)
        self.branches = nn.ModuleList(
            nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks_per_branch)])
            for ch in self.channels
        )

        transforms: list[nn.ModuleList] = []
        for target_index, target_channels in enumerate(self.channels):
            row = nn.ModuleList()
            for source_index, source_channels in enumerate(self.channels):
                if source_index == target_index:
                    row.append(nn.Identity())
                elif source_index > target_index:
                    row.append(ConvNormAct(source_channels, target_channels, 1, activation=False))
                else:
                    row.append(
                        DownsamplePath(
                            source_channels,
                            target_channels,
                            target_index - source_index,
                        )
                    )
            transforms.append(row)
        self.transforms = nn.ModuleList(transforms)
        self.act = nn.GELU()

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        encoded = [branch(feature) for branch, feature in zip(self.branches, features)]
        outputs: list[Tensor] = []
        for target_index, target in enumerate(encoded):
            fused = None
            for source_index, source in enumerate(encoded):
                value = self.transforms[target_index][source_index](source)
                if source_index > target_index:
                    value = F.interpolate(
                        value,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                fused = value if fused is None else fused + value
            outputs.append(self.act(fused))
        return outputs


class NativeJPEGForensicHRNet(nn.Module):
    """W32-scale native-JPEG forensic encoder returning N1--N4."""

    requires_native_jpeg = True

    def __init__(
        self,
        residual_kernels: int = 8,
        hr_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        output_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = hr_channels
        self.residual_bank = ConstrainedResidualBank(residual_kernels)
        self.residual_stem = nn.Sequential(
            ConvNormAct(residual_kernels, c1, stride=2),
            ConvNormAct(c1, c1, stride=2),
            ResidualBlock(c1),
        )

        # Native coefficient grid has one channel for every 8x8 frequency.
        # A second 64-channel plane broadcasts the matching quantization step.
        self.jpeg_stem = nn.Sequential(
            ConvNormAct(128, c1, 3),
            ResidualBlock(c1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.Sigmoid(),
        )
        self.quality_gate = nn.Sequential(
            nn.Linear(4, c1),
            nn.GELU(),
            nn.Linear(c1, c1),
            nn.Sigmoid(),
        )

        self.stage1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))
        self.to_stage2 = ConvNormAct(c1, c2, stride=2)
        self.exchange2 = nn.Sequential(
            MultiResolutionExchange((c1, c2)),
            MultiResolutionExchange((c1, c2)),
        )
        self.to_stage3 = ConvNormAct(c2, c3, stride=2)
        self.exchange3 = nn.Sequential(
            MultiResolutionExchange((c1, c2, c3)),
            MultiResolutionExchange((c1, c2, c3)),
        )
        self.to_stage4 = ConvNormAct(c3, c4, stride=2)
        self.exchange4 = nn.Sequential(
            MultiResolutionExchange((c1, c2, c3, c4)),
            MultiResolutionExchange((c1, c2, c3, c4)),
            MultiResolutionExchange((c1, c2, c3, c4)),
        )
        self.output_adapters = nn.ModuleList(
            ConvNormAct(source, target, 1)
            for source, target in zip(hr_channels, output_channels)
        )

        # Start with a residual-dominant forensic representation.
        nn.init.zeros_(self.spatial_gate[0].weight)
        nn.init.constant_(self.spatial_gate[0].bias, -1.0)
        nn.init.zeros_(self.quality_gate[-2].weight)
        nn.init.constant_(self.quality_gate[-2].bias, 1.0)

    @staticmethod
    def _quality_descriptor(qtable: Tensor) -> Tensor:
        flat = torch.log1p(qtable.float().reshape(qtable.shape[0], -1))
        return torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1),
                flat.amin(dim=1),
                flat.amax(dim=1),
            ],
            dim=1,
        )

    def forward(
        self,
        image: Tensor,
        native_dct: Tensor,
        qtable: Tensor,
        dct_available: Tensor | None = None,
    ) -> list[Tensor]:
        if native_dct.ndim != 4 or native_dct.shape[1] != 64:
            raise ValueError(
                f"native DCT must be [B,64,H/8,W/8], got {tuple(native_dct.shape)}"
            )

        residual = self.residual_stem(self.residual_bank(image))
        coefficients = torch.sign(native_dct.float()) * torch.log1p(native_dct.float().abs())
        qflat = torch.log1p(qtable.float().reshape(qtable.shape[0], 64))
        qplanes = qflat[:, :, None, None].expand_as(coefficients)
        jpeg = self.jpeg_stem(torch.cat([coefficients, qplanes], dim=1).to(image.dtype))
        jpeg = F.interpolate(jpeg, size=residual.shape[-2:], mode="bilinear", align_corners=False)

        spatial = self.spatial_gate(torch.cat([residual, jpeg], dim=1))
        quality = self.quality_gate(self._quality_descriptor(qtable)).to(image.dtype)
        quality = quality[:, :, None, None]
        if dct_available is not None:
            availability = dct_available.reshape(image.shape[0], 1, 1, 1).to(image.dtype)
            quality = quality * availability
        branch1 = self.stage1(residual + spatial * quality * jpeg)

        branch2 = self.to_stage2(branch1)
        branches = self.exchange2([branch1, branch2])
        branch3 = self.to_stage3(branches[1])
        branches = self.exchange3([*branches, branch3])
        branch4 = self.to_stage4(branches[2])
        branches = self.exchange4([*branches, branch4])
        return [adapter(feature) for adapter, feature in zip(self.output_adapters, branches)]

