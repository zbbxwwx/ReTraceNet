"""ReTraceNet model definition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import (
    ConstrainedResidualBank,
    ConvNeXtBlock,
    ContentOrthogonalization,
    DualHypothesisEvidenceLedger,
    EvidenceRecycler,
    IntraImageCounterfactualIntervention,
    LayerNorm2d,
    QualityConditioner,
    QualityDescriptor,
    UncertaintyBoundaryRefiner,
    linear_drop_path_schedule,
    make_stage,
)
from .hierarchy import (
    CoarseToFineICVMPrior,
    ConflictGuidedBoundaryRefinement,
    ConflictConditionedEvidenceReasoning,
    HierarchicalReliabilityPropagation,
    LightweightResidualStream,
    MultiScaleEvidenceFeedback,
    SpatialReliabilityPerception,
)


class ResNetFeaturePyramid(nn.Module):
    """ResNet50 C2-C5 feature pyramid used as a stronger encoder backend."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        conv1_weight = None
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
            conv1_weight = state.get("conv1.weight")
            if in_channels != 3:
                state = {k: v for k, v in state.items() if k != "conv1.weight"}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected:
                print(f"ResNetFeaturePyramid ignored unexpected keys: {len(unexpected)}")
            if len(missing) > (1 if in_channels != 3 else 0):
                print(f"ResNetFeaturePyramid missing keys: {len(missing)}")
        if in_channels != 3:
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            if conv1_weight is not None:
                adapted = conv1_weight.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
                adapted = adapted * (3.0 / float(in_channels))
                backbone.conv1.weight.data.copy_(adapted)
            else:
                nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.imagenet_normalize = bool(imagenet_normalize and in_channels == 3)
        self.freeze_bn = bool(freeze_bn)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if self.freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn()
        return self

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.imagenet_normalize:
            x = (x - self.mean) / self.std
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class ReTraceNet(nn.Module):
    """Counterfactual evidence accounting network for faint traces.

    Inputs are float RGB tensors in [0, 1]. The primary outputs are raw logits;
    sigmoid is deliberately kept outside the model for numerically stable loss.
    """

    def __init__(
        self,
        base_channels: int = 32,
        decoder_channels: int = 96,
        depths: Sequence[int] = (2, 2, 3, 3),
        recycle_steps: int = 2,
        residual_kernels: int = 8,
        drop_path: float = 0.1,
        encoder_backbone: str = "custom",
        pretrained_path: str | None = None,
        freeze_encoder_bn: bool = True,
        use_quality_calibration: bool = True,
        use_content_orthogonalization: bool = True,
        use_counterfactual_intervention: bool = True,
        use_evidence_ledger: bool = True,
        use_uncertainty_refinement: bool = True,
        simple_rgb_trace_baseline: bool = False,
        use_content_decoder_fusion: bool = False,
        decoder_source: str = "trace",
        detach_counterfactual_anchor: bool = False,
        intervention_strength_scale: float = 1.0,
        residual_intervention_output: bool = False,
        residual_intervention_init: float = -4.0,
        intervention_neighborhood: int = 9,
        intervention_gate_mode: str = "probability",
        intervention_gate_threshold: float = 0.2,
        intervention_gate_temperature: float = 0.05,
        use_causal_correction_head: bool = False,
        causal_correction_init: float = -6.0,
        causal_correction_channels: int = 1,
        causal_correction_injection: str = "direct",
        causal_correction_injection_scale: float = 1.0,
        causal_correction_causal_threshold: float = 0.0,
        causal_correction_gate_temperature: float = 0.1,
        causal_correction_gate_init: float = -6.0,
        use_highres_decoder_refinement: bool = False,
        highres_refinement_init: float = -4.0,
        use_dhel_residual_correction: bool = False,
        dhel_residual_init: float = -6.0,
        use_dhel_feature_fusion: bool = False,
        dhel_feature_fusion_init: float = -4.0,
        intervention_teacher_mix: float = 0.5,
        use_residual_stream: bool = False,
        residual_encoder_channels: Sequence[int] = (32, 64, 128, 256),
        residual_encoder_type: str = "lightweight_srm",
        use_cross_scale_reliability: bool = False,
        use_spatial_reliability: bool = False,
        spatial_reliability_hidden: int = 64,
        use_coarse_to_fine_icvm: bool = False,
        use_conflict_cross_scale_reasoning: bool = False,
        use_multiscale_evidence_feedback: bool = False,
        use_reliability_conflict_feedback_gate: bool = False,
        use_cbr: bool = False,
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError("ReTraceNet requires four encoder depths")
        channels = [base_channels * (2**i) for i in range(4)]
        schedule = linear_drop_path_schedule(depths, drop_path)
        self.encoder_backbone = str(encoder_backbone)
        self.use_resnet_encoder = self.encoder_backbone in {"resnet50", "dual_resnet50"}
        if self.use_resnet_encoder and channels != [256, 512, 1024, 2048]:
            raise ValueError("resnet50 encoder requires base_channels=256")
        self.use_quality_calibration = use_quality_calibration
        self.use_content_orthogonalization = use_content_orthogonalization
        self.use_counterfactual_intervention = use_counterfactual_intervention
        self.use_evidence_ledger = use_evidence_ledger
        self.use_uncertainty_refinement = use_uncertainty_refinement
        self.simple_rgb_trace_baseline = simple_rgb_trace_baseline
        self.use_content_decoder_fusion = use_content_decoder_fusion
        self.decoder_source = decoder_source
        if self.decoder_source not in {"trace", "content", "trace_content"}:
            raise ValueError("decoder_source must be one of: trace, content, trace_content")
        self.detach_counterfactual_anchor = bool(detach_counterfactual_anchor)
        self.intervention_strength_scale = float(intervention_strength_scale)
        self.residual_intervention_output = bool(residual_intervention_output)
        self.residual_intervention_gain = nn.Parameter(torch.tensor(float(residual_intervention_init)))
        self.intervention_teacher_mix = float(intervention_teacher_mix)
        self.use_causal_correction_head = bool(use_causal_correction_head)
        self.causal_correction_channels = int(causal_correction_channels)
        self.causal_correction_injection = str(causal_correction_injection)
        self.causal_correction_injection_scale = float(causal_correction_injection_scale)
        self.causal_correction_causal_threshold = float(causal_correction_causal_threshold)
        self.causal_correction_gate_temperature = max(1e-6, float(causal_correction_gate_temperature))
        self.causal_correction_gate_init = float(causal_correction_gate_init)
        self.use_highres_decoder_refinement = bool(use_highres_decoder_refinement)
        self.highres_refinement_gain = nn.Parameter(torch.tensor(float(highres_refinement_init)))
        self.use_dhel_residual_correction = bool(use_dhel_residual_correction)
        self.dhel_residual_gain = nn.Parameter(torch.tensor(float(dhel_residual_init)))
        self.use_dhel_feature_fusion = bool(use_dhel_feature_fusion)
        self.dhel_feature_fusion_gain = nn.Parameter(torch.tensor(float(dhel_feature_fusion_init)))
        self.use_residual_stream = bool(use_residual_stream)
        self.use_cross_scale_reliability = bool(use_cross_scale_reliability)
        self.use_spatial_reliability = bool(use_spatial_reliability)
        self.use_coarse_to_fine_icvm = bool(use_coarse_to_fine_icvm)
        self.use_conflict_cross_scale_reasoning = bool(use_conflict_cross_scale_reasoning)
        self.use_multiscale_evidence_feedback = bool(use_multiscale_evidence_feedback)
        self.use_reliability_conflict_feedback_gate = bool(use_reliability_conflict_feedback_gate)
        self.use_cbr = bool(use_cbr)

        self.residual_bank = ConstrainedResidualBank(residual_kernels)
        self.quality_descriptor = QualityDescriptor()
        self.quality_conditioner = QualityConditioner(channels)
        self.residual_stream = LightweightResidualStream(
            residual_encoder_channels,
            encoder_type=residual_encoder_type,
        )
        self.residual_stream_lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(width), decoder_channels, 1, bias=False),
                    LayerNorm2d(decoder_channels),
                )
                for width in residual_encoder_channels
            ]
        )
        self.residual_stream_gain = nn.Parameter(torch.zeros(4))
        self.cross_scale_reliability = HierarchicalReliabilityPropagation(scales=4)
        self.spatial_reliability = SpatialReliabilityPerception(channels, hidden_channels=spatial_reliability_hidden)

        if self.use_resnet_encoder:
            self.content_encoder = ResNetFeaturePyramid(
                3, pretrained_path=pretrained_path, imagenet_normalize=True, freeze_bn=freeze_encoder_bn
            )
            self.trace_encoder = ResNetFeaturePyramid(
                residual_kernels, pretrained_path=pretrained_path, imagenet_normalize=False, freeze_bn=freeze_encoder_bn
            )
            self.content_stem = nn.Identity()
            self.trace_stem = nn.Identity()
            self.content_down = nn.ModuleList()
            self.trace_down = nn.ModuleList()
            self.content_stages = nn.ModuleList()
            self.trace_stages = nn.ModuleList()
        else:
            self.content_encoder = None
            self.trace_encoder = None
            self.content_stem = nn.Sequential(
                nn.Conv2d(3, channels[0], 5, stride=2, padding=2, bias=False),
                LayerNorm2d(channels[0]),
            )
            self.trace_stem = nn.Sequential(
                nn.Conv2d(residual_kernels, channels[0], 5, stride=2, padding=2, bias=False),
                LayerNorm2d(channels[0]),
            )
            self.content_down = nn.ModuleList()
            self.trace_down = nn.ModuleList()
            self.content_stages = nn.ModuleList()
            self.trace_stages = nn.ModuleList()
            for index, width in enumerate(channels):
                if index > 0:
                    self.content_down.append(
                        nn.Sequential(LayerNorm2d(channels[index - 1]), nn.Conv2d(channels[index - 1], width, 2, 2))
                    )
                    self.trace_down.append(
                        nn.Sequential(LayerNorm2d(channels[index - 1]), nn.Conv2d(channels[index - 1], width, 2, 2))
                    )
                self.content_stages.append(make_stage(width, depths[index], schedule[index]))
                self.trace_stages.append(make_stage(width, depths[index], schedule[index]))
        self.orthogonalizers = nn.ModuleList()
        self.lateral = nn.ModuleList()
        for width in channels:
            self.orthogonalizers.append(ContentOrthogonalization(width))
            self.lateral.append(
                nn.Sequential(nn.Conv2d(width, decoder_channels, 1, bias=False), LayerNorm2d(decoder_channels))
            )
        if self.use_content_decoder_fusion or self.decoder_source in {"content", "trace_content"}:
            self.content_lateral = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(width, decoder_channels, 1, bias=False), LayerNorm2d(decoder_channels))
                    for width in channels
                ]
            )
            self.content_fusion_gain = nn.Parameter(torch.tensor(0.0))
        else:
            self.content_lateral = nn.ModuleList()

        self.highres_decoder_refinement = nn.Sequential(
            nn.Conv2d(3 * decoder_channels, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            ConvNeXtBlock(decoder_channels),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        nn.init.zeros_(self.highres_decoder_refinement[-1].weight)
        nn.init.zeros_(self.highres_decoder_refinement[-1].bias)

        if self.simple_rgb_trace_baseline:
            self.simple_lateral = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(2 * width, decoder_channels, 1, bias=False),
                        LayerNorm2d(decoder_channels),
                        nn.GELU(),
                    )
                    for width in channels
                ]
            )
            self.simple_smooth = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                        LayerNorm2d(decoder_channels),
                        nn.GELU(),
                        nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                        LayerNorm2d(decoder_channels),
                        nn.GELU(),
                    )
                    for _ in channels
                ]
            )
            self.simple_aux_heads = nn.ModuleList([nn.Conv2d(decoder_channels, 1, 1) for _ in channels])
            self.simple_head = nn.Conv2d(decoder_channels, 1, 1)

        self.recycler = EvidenceRecycler(decoder_channels, scales=4, steps=recycle_steps)
        self.counterfactual_intervention = IntraImageCounterfactualIntervention(
            decoder_channels,
            neighborhood=intervention_neighborhood,
            gate_mode=intervention_gate_mode,
            gate_threshold=intervention_gate_threshold,
            gate_temperature=intervention_gate_temperature,
        )
        self.evidence_ledger = DualHypothesisEvidenceLedger(decoder_channels, scales=4)
        self.coarse_to_fine_icvm = CoarseToFineICVMPrior(scales=4)
        self.conflict_cross_scale_reasoning = ConflictConditionedEvidenceReasoning(decoder_channels, scales=4)
        self.multiscale_evidence_feedback = MultiScaleEvidenceFeedback(decoder_channels, scales=4)
        self.cbr = ConflictGuidedBoundaryRefinement(decoder_channels)
        dhel_residual_channels = max(16, decoder_channels // 2)
        self.dhel_residual_head = nn.Sequential(
            nn.Conv2d(5, dhel_residual_channels, 3, padding=1, bias=False),
            LayerNorm2d(dhel_residual_channels),
            nn.GELU(),
            nn.Conv2d(dhel_residual_channels, 1, 1),
        )
        nn.init.zeros_(self.dhel_residual_head[-1].weight)
        nn.init.zeros_(self.dhel_residual_head[-1].bias)
        self.dhel_feature_adapter = nn.Sequential(
            nn.Conv2d(2 * decoder_channels + 4, decoder_channels, 1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 1),
        )
        nn.init.zeros_(self.dhel_feature_adapter[-1].weight)
        nn.init.zeros_(self.dhel_feature_adapter[-1].bias)
        self.aux_heads = nn.ModuleList([nn.Conv2d(decoder_channels, 1, 1) for _ in channels])
        self.refiner = UncertaintyBoundaryRefiner(decoder_channels)
        self.causal_gain = nn.Parameter(torch.tensor(0.0))
        correction_channels = max(16, decoder_channels // 2)
        self.causal_correction_gain = nn.Parameter(torch.tensor(float(causal_correction_init)))
        self.causal_correction_positive_gain = nn.Parameter(torch.tensor(float(causal_correction_init)))
        self.causal_correction_negative_gain = nn.Parameter(torch.tensor(float(causal_correction_init)))
        self.causal_correction_head = nn.Sequential(
            nn.Conv2d(5, correction_channels, 3, padding=1, bias=False),
            LayerNorm2d(correction_channels),
            nn.GELU(),
            nn.Conv2d(correction_channels, self.causal_correction_channels, 1),
        )
        nn.init.zeros_(self.causal_correction_head[-1].weight)
        nn.init.zeros_(self.causal_correction_head[-1].bias)
        if self.causal_correction_injection == "learned_gate":
            self.causal_correction_gate_head = nn.Sequential(
                nn.Conv2d(7, correction_channels, 3, padding=1, bias=False),
                LayerNorm2d(correction_channels),
                nn.GELU(),
                nn.Conv2d(correction_channels, 2, 1),
            )
            nn.init.zeros_(self.causal_correction_gate_head[-1].weight)
            nn.init.constant_(self.causal_correction_gate_head[-1].bias, self.causal_correction_gate_init)
        else:
            self.causal_correction_gate_head = None
        self.image_head = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 1)
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ReTraceNet":
        return cls(**dict(config))

    def forward(
        self, image: Tensor, intervention_mask: Tensor | None = None
    ) -> dict[str, Any]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        input_size = image.shape[-2:]
        descriptor = self.quality_descriptor(image)
        film = self.quality_conditioner(descriptor)
        if self.use_resnet_encoder:
            content_pyramid = self.content_encoder(image)
            trace_pyramid = self.trace_encoder(self.residual_bank(image))
        else:
            content = self.content_stem(image)
            trace = self.trace_stem(self.residual_bank(image))
            content_pyramid = []
            trace_pyramid = []
            for index in range(4):
                if index > 0:
                    content = self.content_down[index - 1](content)
                    trace = self.trace_down[index - 1](trace)
                content = self.content_stages[index](content)
                trace = self.trace_stages[index](trace)
                content_pyramid.append(content)
                trace_pyramid.append(trace)

        trace_features: list[Tensor] = []
        content_features: list[Tensor] = []
        scale_reliability: list[Tensor] = []
        for index, (content, trace) in enumerate(zip(content_pyramid, trace_pyramid)):
            gamma, beta, reliability = film[index]
            if self.use_quality_calibration:
                trace = trace * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            else:
                reliability = torch.ones_like(reliability)
            trace, aligned_content = self.orthogonalizers[index](
                trace, content, enabled=self.use_content_orthogonalization
            )
            trace_features.append(trace)
            content_features.append(aligned_content)
            scale_reliability.append(reliability)

        reliability_messages = [item.new_zeros(item.shape) for item in scale_reliability]
        if self.use_cross_scale_reliability:
            scale_reliability, reliability_messages = self.cross_scale_reliability(scale_reliability)
        spatial_reliability_residuals = [
            feature.new_zeros((feature.shape[0], 1, *feature.shape[-2:]))
            for feature in trace_features
        ]
        if self.use_spatial_reliability:
            scale_reliability, spatial_reliability_residuals = self.spatial_reliability(
                trace_features,
                content_features,
                scale_reliability,
            )
        scale_reliability_vector = torch.cat([
            item.mean((2, 3)) if item.ndim == 4 else item
            for item in scale_reliability
        ], dim=1)
        reliability_message_vector = torch.cat([
            item.mean((2, 3)) if item.ndim == 4 else item
            for item in reliability_messages
        ], dim=1)

        if self.simple_rgb_trace_baseline:
            fused = [
                projection(torch.cat([content_feature, trace_feature], dim=1))
                for projection, content_feature, trace_feature in zip(self.simple_lateral, content_features, trace_features)
            ]
            decoded: list[Tensor] = [fused[-1]] * len(fused)
            state = fused[-1]
            for index in range(len(fused) - 1, -1, -1):
                if index < len(fused) - 1:
                    state = F.interpolate(state, size=fused[index].shape[-2:], mode="bilinear", align_corners=False)
                    state = state + fused[index]
                state = self.simple_smooth[index](state)
                decoded[index] = state
            mask_logits = F.interpolate(self.simple_head(decoded[0]), size=input_size, mode="bilinear", align_corners=False)
            aux_logits = [
                F.interpolate(head(feature), size=input_size, mode="bilinear", align_corners=False)
                for head, feature in zip(self.simple_aux_heads, decoded)
            ]
            zero_map = mask_logits.new_zeros(mask_logits.shape)
            low_zero = decoded[0].new_zeros((decoded[0].shape[0], 1, *decoded[0].shape[-2:]))
            probability = mask_logits.sigmoid()
            pooled = torch.cat([
                probability.mean((2, 3)),
                probability.flatten(1).amax(1, keepdim=True),
            ], dim=1)
            image_logit = self.image_head(pooled)
            return {
                "mask_logits": mask_logits,
                "factual_logits": mask_logits,
                "counterfactual_logits": mask_logits.detach(),
                "causal_effect": zero_map,
                "coarse_logits": mask_logits,
                "proposal_logits": zero_map,
                "log_var": zero_map,
                "image_logit": image_logit,
                "aux_logits": aux_logits,
                "quality_descriptor": descriptor,
                "scale_reliability": scale_reliability_vector,
                "scale_reliability_maps": scale_reliability,
                "spatial_reliability_residuals": spatial_reliability_residuals,
                "hierarchical_reliability_messages": reliability_message_vector,
                "hierarchical_reliability_message_maps": reliability_messages,
                "ledger_logits": [zero_map for _ in decoded],
                "ledger_conflicts": [low_zero for _ in decoded],
                "intervention_effects": [low_zero for _ in decoded],
                "intervention_strengths": [low_zero for _ in decoded],
                "trace_features": trace_features,
                "content_features": content_features,
            }

        trace_anchors = [projection(feature) for projection, feature in zip(self.lateral, trace_features)]
        if self.decoder_source in {"content", "trace_content"} or self.use_content_decoder_fusion:
            content_anchors = [
                projection(feature)
                for projection, feature in zip(self.content_lateral, content_features)
            ]
        else:
            content_anchors = []
        if self.decoder_source == "content":
            anchors = content_anchors
        elif self.decoder_source == "trace_content" or self.use_content_decoder_fusion:
            content_gain = torch.sigmoid(self.content_fusion_gain)
            anchors = [trace_anchor + content_gain * content_anchor for trace_anchor, content_anchor in zip(trace_anchors, content_anchors)]
        else:
            anchors = trace_anchors
        residual_stream_features: list[Tensor] = []
        residual_stream_anchors: list[Tensor] = []
        if self.use_residual_stream:
            residual_stream_features = self.residual_stream(image)
            residual_stream_anchors = [
                projection(feature)
                for projection, feature in zip(self.residual_stream_lateral, residual_stream_features)
            ]
            anchors = [
                anchor + self.residual_stream_gain[index].tanh() * residual_anchor
                for index, (anchor, residual_anchor) in enumerate(zip(anchors, residual_stream_anchors))
            ]
        proposal_logits_low = self.counterfactual_intervention.propose(anchors[0])
        icvm_priors_low: list[Tensor] = [proposal_logits_low]
        icvm_prior_messages: list[Tensor] = [proposal_logits_low.new_zeros(proposal_logits_low.shape)]
        if self.use_coarse_to_fine_icvm:
            local_icvm_priors = [proposal_logits_low] + [
                self.counterfactual_intervention.propose(anchor) for anchor in anchors[1:]
            ]
            icvm_priors_low, icvm_prior_messages = self.coarse_to_fine_icvm(local_icvm_priors)
            proposal_logits_low = icvm_priors_low[0]
        counterfactual_anchors: list[Tensor] = []
        intervention_effects: list[Tensor] = []
        intervention_strengths: list[Tensor] = []
        for anchor in anchors:
            if self.use_counterfactual_intervention:
                counterfactual, effect, strength = self.counterfactual_intervention(
                    anchor,
                    proposal_logits_low,
                    intervention_mask=intervention_mask,
                    teacher_mix=self.intervention_teacher_mix,
                    strength_scale=self.intervention_strength_scale,
                )
                if self.residual_intervention_output:
                    residual_gain = torch.sigmoid(self.residual_intervention_gain)
                    counterfactual = anchor + residual_gain * (counterfactual - anchor)
                    effect = effect * residual_gain
                    strength = strength * residual_gain
                if self.detach_counterfactual_anchor:
                    counterfactual_for_decoder = counterfactual.detach()
                else:
                    counterfactual_for_decoder = counterfactual
            else:
                counterfactual = anchor
                counterfactual_for_decoder = anchor
                effect = anchor.new_zeros((anchor.shape[0], 1, *anchor.shape[-2:]))
                strength = torch.zeros_like(effect)
            counterfactual_anchors.append(counterfactual_for_decoder)
            intervention_effects.append(effect)
            intervention_strengths.append(strength)

        recycled = self.recycler(anchors)
        counterfactual_recycled = (
            self.recycler(counterfactual_anchors)
            if self.use_counterfactual_intervention
            else recycled
        )
        if self.use_evidence_ledger:
            dhel_features, ledger_logits_low, ledger_conflicts, ledger_effects = (
                self.evidence_ledger(
                    recycled, counterfactual_recycled, scale_reliability
                )
            )
            dhel_evidence_features = dhel_features
            ledger_features = recycled if (self.use_dhel_residual_correction or self.use_dhel_feature_fusion) else dhel_features
        else:
            ledger_features = recycled
            dhel_evidence_features = recycled
            ledger_logits_low = [
                state.new_zeros((state.shape[0], 1, *state.shape[-2:])) for state in recycled
            ]
            ledger_conflicts = [
                state.new_zeros((state.shape[0], 1, *state.shape[-2:])) for state in recycled
            ]
            ledger_effects = intervention_effects
        cderf_support_low = [
            state.new_zeros((state.shape[0], 1, *state.shape[-2:])) for state in recycled
        ]
        cderf_refutation_low = [
            state.new_zeros((state.shape[0], 1, *state.shape[-2:])) for state in recycled
        ]
        cderf_conflicts_low = ledger_conflicts
        if self.use_conflict_cross_scale_reasoning:
            dhel_evidence_features, cderf_support_low, cderf_refutation_low, cderf_conflicts_low = (
                self.conflict_cross_scale_reasoning(
                    dhel_evidence_features,
                    ledger_logits_low,
                    ledger_conflicts,
                    scale_reliability,
                )
            )
            if not (self.use_dhel_residual_correction or self.use_dhel_feature_fusion):
                ledger_features = dhel_evidence_features
        dhel_feature_delta_low = recycled[0].new_zeros(recycled[0].shape)
        if self.use_dhel_feature_fusion:
            fusion_size = ledger_features[0].shape[-2:]
            fusion_parts = [
                ledger_features[0],
                dhel_evidence_features[0],
                ledger_logits_low[0],
                ledger_conflicts[0],
                ledger_effects[0],
                intervention_strengths[0],
            ]
            fusion_parts = [
                part if part.shape[-2:] == fusion_size
                else F.interpolate(part, size=fusion_size, mode="bilinear", align_corners=False)
                for part in fusion_parts
            ]
            dhel_feature_delta_low = self.dhel_feature_adapter(torch.cat(fusion_parts, dim=1))
            ledger_features = list(ledger_features)
            ledger_features[0] = ledger_features[0] + torch.sigmoid(self.dhel_feature_fusion_gain) * dhel_feature_delta_low
        feedback_deltas_low = [
            state.new_zeros(state.shape) for state in ledger_features
        ]
        feedback_gates_low = [
            state.new_ones((state.shape[0], 1, *state.shape[-2:])) for state in ledger_features
        ]
        if self.use_multiscale_evidence_feedback:
            ledger_features, feedback_deltas_low, feedback_gates_low = self.multiscale_evidence_feedback(
                ledger_features,
                dhel_evidence_features,
                scale_reliability,
                cderf_conflicts_low,
                use_reliability_conflict_gate=self.use_reliability_conflict_feedback_gate,
            )
        aux_logits = [
            F.interpolate(head(feature), size=input_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.aux_heads, ledger_features)
        ]
        # The causal contrast is measured by a strictly weight-shared verifier;
        # both branches receive identical decoding operations and differ only in
        # the trace intervention.
        factual_verified, _, _ = self.refiner(recycled[0])
        counterfactual_verified, _, _ = self.refiner(counterfactual_recycled[0])
        ledger_refined, log_var, coarse = self.refiner(
            ledger_features[0], ledger_conflicts[0], ledger_effects[0]
        )
        if not self.use_uncertainty_refinement:
            factual_verified = self.refiner.coarse_head(recycled[0])
            counterfactual_verified = self.refiner.coarse_head(counterfactual_recycled[0])
            ledger_refined = coarse
            log_var = torch.zeros_like(log_var)
        causal_effect_low = factual_verified - counterfactual_verified
        dhel_base_logits_low = ledger_refined.detach()
        dhel_residual_logits_low = torch.zeros_like(ledger_refined)
        if self.use_dhel_residual_correction:
            dhel_effect = ledger_effects[0]
            dhel_strength = intervention_strengths[0]
            if dhel_effect.shape[-2:] != ledger_refined.shape[-2:]:
                dhel_effect = F.interpolate(dhel_effect, size=ledger_refined.shape[-2:], mode="bilinear", align_corners=False)
            if dhel_strength.shape[-2:] != ledger_refined.shape[-2:]:
                dhel_strength = F.interpolate(dhel_strength, size=ledger_refined.shape[-2:], mode="bilinear", align_corners=False)
            dhel_residual_input = torch.cat(
                [ledger_logits_low[0], ledger_conflicts[0], dhel_effect, causal_effect_low.detach(), dhel_strength.detach()],
                dim=1,
            )
            dhel_residual_logits_low = self.dhel_residual_head(dhel_residual_input)
            ledger_refined = ledger_refined + torch.sigmoid(self.dhel_residual_gain) * dhel_residual_logits_low

        if self.use_causal_correction_head:
            proposal_probability_low = proposal_logits_low.sigmoid()
            correction_parts = [
                ledger_refined,
                causal_effect_low,
                proposal_probability_low,
                intervention_strengths[0],
                intervention_effects[0],
            ]
            correction_parts = [
                part if part.shape[-2:] == ledger_refined.shape[-2:]
                else F.interpolate(part, size=ledger_refined.shape[-2:], mode="bilinear", align_corners=False)
                for part in correction_parts
            ]
            correction_raw = self.causal_correction_head(torch.cat(correction_parts, dim=1))
            if correction_raw.shape[1] == 2:
                correction_pos_low = correction_raw[:, 0:1]
                correction_neg_low = correction_raw[:, 1:2]
                correction_pos_gate_low = correction_raw.new_full(correction_pos_low.shape, -30.0)
                correction_neg_gate_low = correction_raw.new_full(correction_neg_low.shape, -30.0)
                if self.causal_correction_injection in {"none", "disabled"}:
                    correction_pos_effect = correction_pos_low.new_zeros(correction_pos_low.shape)
                    correction_neg_effect = correction_neg_low.new_zeros(correction_neg_low.shape)
                    correction = correction_pos_effect - correction_neg_effect
                    refined = ledger_refined
                elif self.causal_correction_injection == "learned_gate":
                    gate_inputs = correction_parts + [correction_pos_low, correction_neg_low]
                    gate_logits_low = self.causal_correction_gate_head(torch.cat(gate_inputs, dim=1))
                    correction_pos_gate_low = gate_logits_low[:, 0:1]
                    correction_neg_gate_low = gate_logits_low[:, 1:2]
                    correction_pos_effect = correction_pos_gate_low.sigmoid() * correction_pos_low.sigmoid()
                    correction_neg_effect = correction_neg_gate_low.sigmoid() * correction_neg_low.sigmoid()
                    correction = self.causal_correction_injection_scale * (correction_pos_effect - correction_neg_effect)
                elif self.causal_correction_injection == "selective":
                    base_probability_low = ledger_refined.detach().sigmoid()
                    causal_support_gate = torch.sigmoid(
                        (causal_effect_low.detach() - self.causal_correction_causal_threshold)
                        / self.causal_correction_gate_temperature
                    )
                    causal_refute_gate = 1.0 - causal_support_gate
                    positive_gate = correction_pos_low.sigmoid() * causal_support_gate * (1.0 - base_probability_low)
                    negative_gate = correction_neg_low.sigmoid() * causal_refute_gate * base_probability_low
                    correction_pos_effect = positive_gate * F.softplus(correction_pos_low)
                    correction_neg_effect = negative_gate * F.softplus(correction_neg_low)
                    correction = self.causal_correction_injection_scale * (correction_pos_effect - correction_neg_effect)
                else:
                    correction_pos_effect = correction_pos_low
                    correction_neg_effect = correction_neg_low
                    correction = correction_pos_low - correction_neg_low
                if self.causal_correction_injection not in {"none", "disabled"}:
                    refined = (
                        ledger_refined
                        + torch.sigmoid(self.causal_correction_positive_gain)
                        * self.causal_correction_injection_scale
                        * correction_pos_effect
                        - torch.sigmoid(self.causal_correction_negative_gain)
                        * self.causal_correction_injection_scale
                        * correction_neg_effect
                    )
            else:
                correction_pos_low = correction_raw.new_zeros(correction_raw.shape)
                correction_neg_low = correction_raw.new_zeros(correction_raw.shape)
                correction_pos_gate_low = correction_raw.new_full(correction_raw.shape, -30.0)
                correction_neg_gate_low = correction_raw.new_full(correction_raw.shape, -30.0)
                correction = correction_raw
                refined = ledger_refined + torch.sigmoid(self.causal_correction_gain) * correction
            correction_logits_low = ledger_refined.detach() + correction
        else:
            correction_logits_low = None
            correction_pos_low = None
            correction_neg_low = None
            correction_pos_gate_low = None
            correction_neg_gate_low = None
            if self.use_dhel_residual_correction or self.use_dhel_feature_fusion:
                refined = ledger_refined
            else:
                refined = ledger_refined + torch.sigmoid(self.causal_gain) * F.relu(causal_effect_low)
        highres_refinement_logits_low = torch.zeros_like(refined)
        if self.use_highres_decoder_refinement:
            c3_refined = F.interpolate(
                ledger_features[1], size=ledger_features[0].shape[-2:], mode="bilinear", align_corners=False
            )
            highres_input = torch.cat([ledger_features[0], anchors[0], c3_refined], dim=1)
            highres_refinement_logits_low = self.highres_decoder_refinement(highres_input)
            refined = refined + torch.sigmoid(self.highres_refinement_gain) * highres_refinement_logits_low
        cbr_delta_low = torch.zeros_like(refined)
        cbr_gate_low = torch.zeros_like(refined)
        cbr_boundary_low = torch.zeros_like(refined)
        if self.use_cbr:
            refined, cbr_delta_low, cbr_gate_low, cbr_boundary_low = self.cbr(
                ledger_features[0],
                refined,
                cderf_conflicts_low[0],
            )
        mask_logits = F.interpolate(refined, size=input_size, mode="bilinear", align_corners=False)
        highres_refinement_logits = F.interpolate(
            highres_refinement_logits_low, size=input_size, mode="bilinear", align_corners=False
        )
        dhel_feature_delta = F.interpolate(
            dhel_feature_delta_low.mean(dim=1, keepdim=True), size=input_size, mode="bilinear", align_corners=False
        )
        dhel_residual_logits = F.interpolate(
            dhel_residual_logits_low, size=input_size, mode="bilinear", align_corners=False
        )
        dhel_base_logits = F.interpolate(
            dhel_base_logits_low, size=input_size, mode="bilinear", align_corners=False
        )
        cbr_delta = F.interpolate(cbr_delta_low, size=input_size, mode="bilinear", align_corners=False)
        cbr_gate = F.interpolate(cbr_gate_low, size=input_size, mode="bilinear", align_corners=False)
        cbr_boundary = F.interpolate(cbr_boundary_low, size=input_size, mode="bilinear", align_corners=False)
        if correction_logits_low is not None:
            base_logits = F.interpolate(ledger_refined.detach(), size=input_size, mode="bilinear", align_corners=False)
            correction_delta = F.interpolate(correction, size=input_size, mode="bilinear", align_corners=False)
            correction_logits = F.interpolate(correction_logits_low, size=input_size, mode="bilinear", align_corners=False)
            correction_pos_logits = F.interpolate(correction_pos_low, size=input_size, mode="bilinear", align_corners=False)
            correction_neg_logits = F.interpolate(correction_neg_low, size=input_size, mode="bilinear", align_corners=False)
            correction_pos_gate_logits = F.interpolate(correction_pos_gate_low, size=input_size, mode="bilinear", align_corners=False)
            correction_neg_gate_logits = F.interpolate(correction_neg_gate_low, size=input_size, mode="bilinear", align_corners=False)
        else:
            base_logits = mask_logits.detach()
            correction_delta = torch.zeros_like(mask_logits)
            correction_logits = mask_logits.detach()
            correction_pos_logits = torch.zeros_like(mask_logits)
            correction_neg_logits = torch.zeros_like(mask_logits)
            correction_pos_gate_logits = torch.full_like(mask_logits, -30.0)
            correction_neg_gate_logits = torch.full_like(mask_logits, -30.0)
        factual_logits = F.interpolate(
            factual_verified, size=input_size, mode="bilinear", align_corners=False
        )
        counterfactual_logits = F.interpolate(
            counterfactual_verified, size=input_size, mode="bilinear", align_corners=False
        )
        causal_effect = F.interpolate(
            causal_effect_low, size=input_size, mode="bilinear", align_corners=False
        )
        log_var = F.interpolate(log_var, size=input_size, mode="bilinear", align_corners=False)
        coarse_logits = F.interpolate(coarse, size=input_size, mode="bilinear", align_corners=False)
        proposal_logits = F.interpolate(
            proposal_logits_low, size=input_size, mode="bilinear", align_corners=False
        )
        ledger_logits = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in ledger_logits_low
        ]
        cderf_support = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in cderf_support_low
        ]
        cderf_refutation = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in cderf_refutation_low
        ]
        cderf_conflicts = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in cderf_conflicts_low
        ]
        feedback_delta = [
            F.interpolate(item.mean(dim=1, keepdim=True), size=input_size, mode="bilinear", align_corners=False)
            for item in feedback_deltas_low
        ]
        feedback_gate = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in feedback_gates_low
        ]
        icvm_prior_logits = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in icvm_priors_low
        ]
        icvm_prior_delta = [
            F.interpolate(item, size=input_size, mode="bilinear", align_corners=False)
            for item in icvm_prior_messages
        ]

        probability = mask_logits.sigmoid()
        reliability = torch.exp(-log_var).clamp(0.0, 10.0)
        reliable_mean = (probability * reliability).sum((2, 3)) / (reliability.sum((2, 3)) + 1e-6)
        reliable_max = (probability * reliability).flatten(1).amax(1, keepdim=True)
        image_logit = self.image_head(torch.cat([reliable_mean, reliable_max], dim=1))

        outputs = {
            "mask_logits": mask_logits,
            "factual_logits": factual_logits,
            "counterfactual_logits": counterfactual_logits,
            "causal_effect": causal_effect,
            "coarse_logits": coarse_logits,
            "correction_logits": correction_logits,
            "base_logits": base_logits,
            "correction_delta": correction_delta,
            "correction_pos_logits": correction_pos_logits,
            "correction_neg_logits": correction_neg_logits,
            "correction_pos_gate_logits": correction_pos_gate_logits,
            "correction_neg_gate_logits": correction_neg_gate_logits,
            "highres_refinement_logits": highres_refinement_logits,
            "dhel_feature_delta": dhel_feature_delta,
            "dhel_residual_logits": dhel_residual_logits,
            "dhel_base_logits": dhel_base_logits,
            "cbr_delta": cbr_delta,
            "cbr_gate": cbr_gate,
            "cbr_boundary": cbr_boundary,
            "proposal_logits": proposal_logits,
            "log_var": log_var,
            "image_logit": image_logit,
            "aux_logits": aux_logits,
            "quality_descriptor": descriptor,
            "scale_reliability": scale_reliability_vector,
            "scale_reliability_maps": scale_reliability,
            "spatial_reliability_residuals": spatial_reliability_residuals,
            "hierarchical_reliability_messages": reliability_message_vector,
            "hierarchical_reliability_message_maps": reliability_messages,
            "ledger_logits": ledger_logits,
            "ledger_conflicts": ledger_conflicts,
            "cderf_support": cderf_support,
            "cderf_refutation": cderf_refutation,
            "cderf_conflicts": cderf_conflicts,
            "evidence_feedback_delta": feedback_delta,
            "evidence_feedback_gate": feedback_gate,
            "icvm_prior_logits": icvm_prior_logits,
            "icvm_prior_delta": icvm_prior_delta,
            "intervention_effects": intervention_effects,
            "intervention_strengths": intervention_strengths,
            "trace_features": trace_features,
            "content_features": content_features,
            "residual_stream_features": residual_stream_features,
            "residual_stream_anchors": residual_stream_anchors,
        }
        return outputs


class ResNet50FpnSanity(nn.Module):
    """Plain RGB ResNet50-FPN segmentation sanity baseline."""

    def __init__(
        self,
        pretrained_path: str | None = None,
        decoder_channels: int = 128,
        freeze_bn: bool = True,
        imagenet_normalize: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            if unexpected:
                print(f"ResNet50FpnSanity ignored unexpected keys: {len(unexpected)}")
            if len(missing) > 5:
                print(f"ResNet50FpnSanity missing keys: {len(missing)}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        if freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        widths = [256, 512, 1024, 2048]
        self.lateral = nn.ModuleList([nn.Conv2d(width, decoder_channels, 1) for width in widths])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                LayerNorm2d(decoder_channels),
                nn.GELU(),
            )
            for _ in widths
        ])
        self.aux_heads = nn.ModuleList([nn.Conv2d(decoder_channels, 1, 1) for _ in widths])
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.image_head = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.imagenet_normalize = bool(imagenet_normalize)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, image: Tensor, intervention_mask: Tensor | None = None) -> dict[str, Any]:
        input_size = image.shape[-2:]
        x = (image - self.mean) / self.std if self.imagenet_normalize else image
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = [c2, c3, c4, c5]
        lateral = [proj(feat) for proj, feat in zip(self.lateral, feats)]
        decoded: list[Tensor] = [lateral[-1]] * len(lateral)
        state = lateral[-1]
        for index in range(len(lateral) - 1, -1, -1):
            if index < len(lateral) - 1:
                state = F.interpolate(state, size=lateral[index].shape[-2:], mode="bilinear", align_corners=False)
                state = state + lateral[index]
            state = self.smooth[index](state)
            decoded[index] = state
        mask_logits = F.interpolate(self.head(decoded[0]), size=input_size, mode="bilinear", align_corners=False)
        aux_logits = [
            F.interpolate(head(feature), size=input_size, mode="bilinear", align_corners=False)
            for head, feature in zip(self.aux_heads, decoded)
        ]
        zero = mask_logits.new_zeros(mask_logits.shape)
        low_zero = decoded[0].new_zeros((decoded[0].shape[0], 1, *decoded[0].shape[-2:]))
        probability = mask_logits.sigmoid()
        image_logit = self.image_head(torch.cat([probability.mean((2, 3)), probability.flatten(1).amax(1, keepdim=True)], dim=1))
        return {
            "mask_logits": mask_logits,
            "factual_logits": mask_logits,
            "counterfactual_logits": mask_logits.detach(),
            "causal_effect": zero,
            "coarse_logits": mask_logits,
            "proposal_logits": zero,
            "log_var": zero,
            "image_logit": image_logit,
            "aux_logits": aux_logits,
            "quality_descriptor": image.new_zeros((image.shape[0], 2)),
            "scale_reliability": image.new_ones((image.shape[0], 4)),
            "ledger_logits": [zero for _ in decoded],
            "ledger_conflicts": [low_zero for _ in decoded],
            "intervention_effects": [low_zero for _ in decoded],
            "intervention_strengths": [low_zero for _ in decoded],
            "trace_features": decoded,
            "content_features": [feature.detach() for feature in decoded],
        }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    config = dict(config)
    architecture = str(config.pop("architecture", config.pop("type", "retrace")))
    if architecture in {"resnet50_fpn_sanity", "rgb_resnet50_fpn_sanity"}:
        return ResNet50FpnSanity(**config)
    return ReTraceNet.from_config(config)
