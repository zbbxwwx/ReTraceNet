"""Score-first high-resolution FPN decoder on the proven ResNet50 feature basis."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import ConvNeXtBlock
from .backbone import ResNet50ContentEncoder
from .decoder import ProgressiveDecoder
from .forensic_stream import LightweightForensicStream


def _gn(channels: int) -> nn.GroupNorm:
    groups = 32 if channels % 32 == 0 else 16 if channels % 16 == 0 else 8
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            _gn(out_channels),
            nn.GELU(),
        )


class PyramidPoolingContext(nn.Module):
    def __init__(self, width: int = 192, pool_scales: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        branch = max(24, width // 4)
        self.pool_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(width, branch, 1, bias=False), _gn(branch), nn.GELU()) for _ in pool_scales)
        self.fuse = ConvGNAct(width + branch * len(pool_scales), width, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        parts = [x]
        for scale, proj in zip(self.pool_scales, self.pool_proj):
            pooled = F.adaptive_avg_pool2d(x, (scale, scale))
            parts.append(F.interpolate(proj(pooled), size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(parts, dim=1))


class ResidualRefineBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvGNAct(width, width, 3), ConvNeXtBlock(width))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class HighResolutionFPNDecoder(nn.Module):
    def __init__(
        self,
        content_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        forensic_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        width: int = 192,
    ) -> None:
        super().__init__()
        self.width = width
        self.proj = nn.ModuleList(
            ConvGNAct(c + n, width, 1) for c, n in zip(content_channels, forensic_channels)
        )
        self.context = PyramidPoolingContext(width)
        self.refine3 = ConvGNAct(width, width, 3)
        self.refine2 = ConvGNAct(width, width, 3)
        self.refine1 = ConvGNAct(width, width, 3)
        self.fusion = ConvGNAct(width * 4, width, 3)
        self.res_refine1 = ResidualRefineBlock(width)
        self.res_refine2 = ResidualRefineBlock(width)

    @staticmethod
    def _resize_like(x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, content: list[Tensor], detail: list[Tensor]) -> dict[str, list[Tensor] | Tensor]:
        feats = []
        for c, n, proj in zip(content, detail, self.proj):
            if c.shape[-2:] != n.shape[-2:]:
                c = F.interpolate(c, size=n.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(proj(torch.cat([c, n], dim=1)))
        f1, f2, f3, f4 = feats
        p4 = self.context(f4)
        p3 = self.refine3(f3 + self._resize_like(p4, f3))
        p2 = self.refine2(f2 + self._resize_like(p3, f2))
        p1 = self.refine1(f1 + self._resize_like(p2, f1))
        fused = torch.cat([
            p1,
            self._resize_like(p2, p1),
            self._resize_like(p3, p1),
            self._resize_like(p4, p1),
        ], dim=1)
        out = self.fusion(fused)
        out = self.res_refine1(out)
        out = self.res_refine2(out)
        return {"pyramid": [p1, p2, p3, p4], "decoded": out}


class ScoreModelV2(nn.Module):
    def __init__(
        self,
        decoder_width: int = 192,
        residual_kernels: int = 8,
        forensic_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        pretrained_path: str | None = None,
        freeze_encoder_bn: bool = True,
    ) -> None:
        super().__init__()
        self.content_encoder = ResNet50ContentEncoder(pretrained_path=pretrained_path, freeze_bn=freeze_encoder_bn)
        self.forensic_stream = LightweightForensicStream(residual_kernels=residual_kernels, channels=tuple(forensic_channels))
        self.teacher_decoder = ProgressiveDecoder(detail_channels=tuple(forensic_channels), decoder_channels=128)
        self.teacher_mask_head = nn.Conv2d(128, 1, 1)
        self.fpn_decoder = HighResolutionFPNDecoder(
            content_channels=ResNet50ContentEncoder.channels,
            forensic_channels=tuple(forensic_channels),
            width=decoder_width,
        )
        self.mask_head = nn.Conv2d(decoder_width, 1, 1)
        self.log_var_head = nn.Conv2d(decoder_width, 1, 1)
        self.aux_heads = nn.ModuleList(nn.Conv2d(decoder_width, 1, 1) for _ in range(3))
        self.image_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(decoder_width, 1))

    def freeze_teacher(self) -> None:
        for module in (self.teacher_decoder, self.teacher_mask_head):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def _compat(self, logits: Tensor, log_var: Tensor, image_logit: Tensor, aux: list[Tensor], content: list[Tensor]) -> dict[str, Any]:
        zeros = torch.zeros_like(logits)
        coarse = F.interpolate(logits, size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        strength = F.interpolate(logits.sigmoid().detach(), size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        out: dict[str, Any] = {
            "mask_logits": logits,
            "base_logits": logits,
            "proposal_logits": logits,
            "coarse_logits": coarse,
            "factual_logits": logits,
            "counterfactual_logits": logits.detach(),
            "log_var": log_var,
            "image_logit": image_logit,
            "aux_logits": aux,
            "content_features": content,
            "trace_features": content,
            "quality_descriptor": logits.new_zeros((logits.shape[0], 8)),
            "scale_reliability": [logits.new_full((logits.shape[0], 1, 1, 1), 0.5) for _ in range(4)],
            "ledger_logits": [logits],
            "ledger_conflicts": [zeros],
            "causal_effect": zeros,
            "intervention_effects": [strength],
            "intervention_strengths": [strength],
        }
        return out

    def teacher_logits(self, content: list[Tensor], detail: list[Tensor], image_size: tuple[int, int]) -> Tensor:
        self.freeze_teacher()
        with torch.no_grad():
            decoded = self.teacher_decoder(content, detail)
            logits_low = self.teacher_mask_head(decoded[0])
            return F.interpolate(logits_low, size=image_size, mode="bilinear", align_corners=False)

    def forward(self, image: Tensor, intervention_mask: Tensor | None = None, return_teacher: bool = False) -> dict[str, Any]:
        _ = intervention_mask
        content = self.content_encoder(image)
        detail = self.forensic_stream(image)
        dec = self.fpn_decoder(content, detail)
        p1, p2, p3, _p4 = dec["pyramid"]
        decoded = dec["decoded"]
        logits_low = self.mask_head(decoded)
        logits = F.interpolate(logits_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        log_var = F.interpolate(self.log_var_head(decoded), size=image.shape[-2:], mode="bilinear", align_corners=False)
        aux_low = [head(feat) for head, feat in zip(self.aux_heads, [p1, p2, p3])]
        aux = [F.interpolate(a, size=image.shape[-2:], mode="bilinear", align_corners=False) for a in aux_low]
        out = self._compat(logits, log_var, self.image_head(decoded), aux, content)
        out["aux_logits_low"] = aux_low
        out["fpn_pyramid"] = [p1, p2, p3, _p4]
        if return_teacher:
            out["teacher_logits"] = self.teacher_logits(content, detail, image.shape[-2:])
        return out


def build_score_model_v2(config: dict) -> ScoreModelV2:
    return ScoreModelV2(
        decoder_width=int(config.get("decoder_width", config.get("decoder_channels", 192))),
        residual_kernels=int(config.get("residual_kernels", 8)),
        forensic_channels=tuple(config.get("forensic_channels", (64, 128, 256, 512))),
        pretrained_path=config.get("pretrained_path"),
        freeze_encoder_bn=bool(config.get("freeze_encoder_bn", True)),
    )
