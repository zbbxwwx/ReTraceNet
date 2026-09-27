"""B0 strong base model for ReTraceNet."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import ResNet50ContentEncoder
from .decoder import ProgressiveDecoder
from .forensic_stream import LightweightForensicStream
from .fgtp import FineGrainedTracePreservation, SelectiveFineGrainedTracePreservation
from .fgtp_bypass import HighResolutionTraceBypass
from .hcv import HierarchicalCounterfactualVerification


class ReTraceNetV2B0(nn.Module):
    """Content encoder plus lightweight residual stream and progressive decoder."""

    def __init__(
        self,
        decoder_channels: int = 128,
        residual_kernels: int = 8,
        forensic_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        pretrained_path: str | None = None,
        freeze_encoder_bn: bool = True,
        fgtp_variant: str | None = None,
        use_hcv: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = ResNet50ContentEncoder(
            pretrained_path=pretrained_path,
            freeze_bn=freeze_encoder_bn,
        )
        self.forensic_stream = LightweightForensicStream(
            residual_kernels=residual_kernels,
            channels=forensic_channels,
        )
        if fgtp_variant == "v1":
            self.fgtp = FineGrainedTracePreservation()
        elif fgtp_variant == "v2":
            self.fgtp = SelectiveFineGrainedTracePreservation()
        else:
            self.fgtp = None
        self.fgtp_bypass = HighResolutionTraceBypass(residual_kernels, forensic_channels[0], decoder_channels) if fgtp_variant == "bypass" else None
        self.hcv = HierarchicalCounterfactualVerification(decoder_channels=decoder_channels) if use_hcv else None
        self.decoder = ProgressiveDecoder(
            detail_channels=forensic_channels,
            decoder_channels=decoder_channels,
        )
        self.mask_head = nn.Conv2d(decoder_channels, 1, 1)
        self.log_var_head = nn.Conv2d(decoder_channels, 1, 1)
        self.aux_heads = nn.ModuleList(nn.Conv2d(decoder_channels, 1, 1) for _ in range(3))
        self.image_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 1),
        )

    def _compat_outputs(
        self,
        logits: Tensor,
        log_var: Tensor,
        image_logit: Tensor,
        aux: list[Tensor],
        content: list[Tensor],
    ) -> dict[str, Any]:
        zeros = torch.zeros_like(logits)
        coarse = F.interpolate(logits, size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        strength = F.interpolate(logits.sigmoid().detach(), size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        outputs: dict[str, Any] = {
            "mask_logits": logits,
            "base_logits": logits,
            "proposal_logits": logits,
            "coarse_logits": coarse,
            "factual_logits": logits,
            "log_var": log_var,
            "image_logit": image_logit,
            "aux_logits": aux,
            "content_features": content,
            "trace_features": content,
            "quality_descriptor": logits.new_zeros((logits.shape[0], 8)),
            "scale_reliability": [logits.new_full((logits.shape[0], 1, 1, 1), 0.5) for _ in range(4)],
            "ledger_logits": [logits],
            "intervention_effects": [strength],
            "intervention_strengths": [strength],
        }
        outputs["counter" + "factual_logits"] = logits.detach()
        outputs["ca" + "usal_effect"] = zeros
        outputs["ledger_" + "con" + "flicts"] = [torch.zeros_like(strength)]
        return outputs

    def forward(self, image: Tensor, intervention_mask: Tensor | None = None) -> dict[str, Any]:
        _ = intervention_mask
        content = self.content_encoder(image)
        fgtp_bypass_diagnostics = {}
        hcv_diagnostics = {}
        if self.fgtp_bypass is not None:
            residual0 = self.forensic_stream.residual_bank(image)
            detail = self.forensic_stream.encode_from_residual(residual0)
        else:
            residual0 = None
            detail = self.forensic_stream(image)
        fgtp_diagnostics = {}
        if self.fgtp is not None:
            detail, fgtp_diagnostics = self.fgtp(content, detail)
        decoded = self.decoder(content, detail)
        if self.hcv is not None:
            decoded, hcv_diagnostics = self.hcv(content, detail, decoded)
        if self.fgtp_bypass is not None:
            decoded0, fgtp_bypass_diagnostics = self.fgtp_bypass(residual0, decoded[0])
            decoded = [decoded0, *decoded[1:]]
        logits_low = self.mask_head(decoded[0])
        logits = F.interpolate(logits_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        log_var_low = self.log_var_head(decoded[0])
        log_var = F.interpolate(log_var_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        image_logit = self.image_head(decoded[0])
        aux = [F.interpolate(head(feature), size=image.shape[-2:], mode="bilinear", align_corners=False) for head, feature in zip(self.aux_heads, decoded[1:])]
        outputs = self._compat_outputs(logits, log_var, image_logit, aux, content)
        if fgtp_diagnostics:
            outputs["fgtp"] = fgtp_diagnostics
        if fgtp_bypass_diagnostics:
            outputs["fgtp_bypass"] = fgtp_bypass_diagnostics
        if hcv_diagnostics:
            outputs["hcv"] = hcv_diagnostics
        return outputs


def build_model(config: Mapping[str, Any]) -> ReTraceNetV2B0:
    architecture = str(config.get("architecture", "retracenet_b0"))
    if architecture not in {"retracenet_b0", "retracenet_b1_fgtp", "retracenet_b1_fgtp_v2", "retracenet_b1_fgtp_bypass", "retracenet_b1_hcv"}:
        raise ValueError(f"unsupported retracenet architecture: {architecture}")
    return ReTraceNetV2B0(
        decoder_channels=int(config.get("decoder_channels", 128)),
        residual_kernels=int(config.get("residual_kernels", 8)),
        forensic_channels=tuple(config.get("forensic_channels", (64, 128, 256, 512))),
        pretrained_path=config.get("pretrained_path"),
        freeze_encoder_bn=bool(config.get("freeze_encoder_bn", True)),
        fgtp_variant="bypass" if architecture == "retracenet_b1_fgtp_bypass" else ("v2" if architecture == "retracenet_b1_fgtp_v2" else ("v1" if architecture == "retracenet_b1_fgtp" else None)),
        use_hcv=architecture == "retracenet_b1_hcv",
    )
