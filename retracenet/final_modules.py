"""Final overnight QTR/HCV/DHER modules for ReTraceNet."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import ConvNeXtBlock, LayerNorm2d
from .backbone import ConvNeXtTinyContentEncoder, HRNetW18ContentEncoder, ResNeXt101_32x8dContentEncoder, ResNet101ContentEncoder, ResNet50ContentEncoder
from .decoder import ProgressiveDecoder
from .forensic_stream import LightweightForensicStream, SRMGuidedForensicStream
from .compression_dct_encoder import DCTCueEncoder
from .native_jpeg_hrnet import NativeJPEGForensicHRNet


def norm2d(x: Tensor, eps: float = 1e-6) -> Tensor:
    m = x.mean(1, keepdim=True)
    v = (x - m).square().mean(1, keepdim=True)
    return (x - m) * torch.rsqrt(v + eps)


def dice_loss_from_prob(prob: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    den = prob.sum(1) + target.sum(1)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def gray_quality_descriptor(image: Tensor) -> Tensor:
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mean = gray.mean((2, 3))
    std = gray.std((2, 3), unbiased=False)
    sat = ((image < 0.02) | (image > 0.98)).float().mean((1, 2, 3), keepdim=False)[:, None]
    gx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean((2, 3))
    gy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean((2, 3))
    lap = F.conv2d(gray, gray.new_tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]), padding=1).abs().mean((2, 3))
    if gray.shape[-1] > 8:
        right = gray[:, :, :, 8::8]
        left = gray[:, :, :, 7:-1:8]
        bx = (right - left).abs().mean((2, 3)) if right.numel() > 0 and right.shape == left.shape else gx.new_zeros(gx.shape)
    else:
        bx = gx.new_zeros(gx.shape)
    if gray.shape[-2] > 8:
        bottom = gray[:, :, 8::8, :]
        top = gray[:, :, 7:-1:8, :]
        by = (bottom - top).abs().mean((2, 3)) if bottom.numel() > 0 and bottom.shape == top.shape else gy.new_zeros(gy.shape)
    else:
        by = gy.new_zeros(gy.shape)
    block = 0.5 * (bx + by)
    return torch.cat([mean, std, sat, gx, gy, lap, block], dim=1)


class BalancedForensicFusion(nn.Module):
    """Equalized residual/DCT fusion; DCT availability is sample-wise."""
    def __init__(self, channels: int) -> None:
        super().__init__(); g=max(1,min(32,channels//4))
        while channels % g: g-=1
        self.residual=nn.Sequential(nn.Conv2d(channels,channels,1,bias=False),nn.GroupNorm(g,channels),nn.GELU())
        self.dct=nn.Sequential(nn.Conv2d(channels,channels,1,bias=False),nn.GroupNorm(g,channels),nn.GELU())
        self.fuse=nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1,bias=False),nn.GroupNorm(g,channels),nn.GELU())
    def forward(self,residual:Tensor,dct:Tensor,available:Tensor):
        if dct.shape[-2:]!=residual.shape[-2:]: dct=F.interpolate(dct,size=residual.shape[-2:],mode="bilinear",align_corners=False)
        r=self.residual(residual); d=self.dct(dct)*available[:,:,None,None]
        return self.fuse(torch.cat([r,d],1)),r,d


class UnifiedForensicStream(nn.Module):
    """Residual Cue + DCT Cue + scale-wise Cue Fusion."""

    def __init__(
        self, residual_encoder, dct_encoder,
        cue_fusions, cue_adapters, evidence_head=None
    ):
        super().__init__()
        self.residual_encoder = residual_encoder
        self.dct_encoder = dct_encoder
        self.cue_fusions = cue_fusions
        self.cue_adapters = cue_adapters
        self.evidence_head = evidence_head

    @property
    def fft_prior_fusion(self):
        return getattr(
            self.residual_encoder, "fft_prior_fusion", None
        )

    @property
    def srm_prior_fusion(self):
        return getattr(
            self.residual_encoder, "srm_prior_fusion", None
        )

    def forward(
        self, image, dct, qtable,
        available=None, disable_dct_evidence=False
    ):
        if dct is None or qtable is None:
            raise ValueError("DCT and qtable required")

        residual = self.residual_encoder(image)
        compression = self.dct_encoder(dct, qtable)

        if len(residual) != 4 or len(compression) != 4:
            raise RuntimeError("Expected four scales")

        if available is None:
            available = dct.new_ones((dct.shape[0], 1))

        available = available.to(
            device=residual[0].device,
            dtype=residual[0].dtype
        )

        if disable_dct_evidence:
            available = torch.zeros_like(available)

        fused, writes, gates = [], [], []

        for r, c, fusion in zip(
            residual, compression, self.cue_fusions
        ):
            y, _, _ = fusion(r, c, available)
            fused.append(y)
            writes.append(y - r)
            gates.append(available[:, :, None, None])

        diag = {
            "available_mean": available.float().mean(),
            "available_count": available.float().sum(),
            "feature_rms": [
                x.detach().float().square().mean().sqrt()
                for x in compression
            ],
            "write_rms": [
                x.detach().float().square().mean().sqrt()
                for x in writes
            ],
            "gate_mean": [
                x.detach().float().mean()
                for x in gates
            ],
            "adapter_weight_rms": [
                a.proj.weight.detach().float()
                .square().mean().sqrt()
                for a in self.cue_adapters
            ],
            "disabled": torch.tensor(
                float(disable_dct_evidence),
                device=residual[0].device
            ),
        }

        if self.evidence_head is not None:
            diag["evidence_logits_low"] = (
                self.evidence_head(compression[0])
            )

        return residual, fused, diag, compression

class QualityTraceQualification(nn.Module):
    def __init__(self, content_channels=(256, 512, 1024), detail_channels=(64, 128, 256), embed_dim: int = 32, use_reliability_gate: bool = False, residual_refinement: bool = False) -> None:
        super().__init__()
        self.use_reliability_gate = bool(use_reliability_gate)
        self.residual_refinement = bool(residual_refinement)
        self.quality_mlp = nn.Sequential(nn.Linear(7, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.content_align = nn.ModuleList([nn.Sequential(nn.Conv2d(c, n, 1, bias=False), LayerNorm2d(n)) for c, n in zip(content_channels, detail_channels)])
        self.qualifiers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n * 3 + embed_dim, n, 1, bias=False), LayerNorm2d(n), nn.GELU(),
                nn.Conv2d(n, n, 3, padding=1, groups=n, bias=False), LayerNorm2d(n), nn.GELU(),
                nn.Conv2d(n, n, 1, bias=False), LayerNorm2d(n), nn.GELU(),
            ) for n in detail_channels
        ])
        self.reliability_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n * 2 + embed_dim, n, 1, bias=False), LayerNorm2d(n), nn.GELU(),
                nn.Conv2d(n, n, 3, padding=1, bias=False), LayerNorm2d(n), nn.GELU(),
                nn.Conv2d(n, 1, 1),
            ) for n in detail_channels
        ]) if self.use_reliability_gate else nn.ModuleList()
        if self.use_reliability_gate:
            for gate in self.reliability_gates:
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)

    def forward(self, image: Tensor, content: list[Tensor], detail: list[Tensor]) -> dict[str, list[Tensor] | Tensor]:
        q = self.quality_mlp(gray_quality_descriptor(image))
        feats, maps, reliability_maps, qualified_detail = [], [], [], []
        for i in range(3):
            c = self.content_align[i](content[i])
            if c.shape[-2:] != detail[i].shape[-2:]:
                c = F.interpolate(c, size=detail[i].shape[-2:], mode="bilinear", align_corners=False)
            qi = q[:, :, None, None].expand(-1, -1, detail[i].shape[-2], detail[i].shape[-1])
            x = torch.cat([detail[i], c, (norm2d(detail[i]) - norm2d(c)).abs(), qi], dim=1)
            f = self.qualifiers[i](x)
            feats.append(f)
            maps.append(f.abs().mean(1, keepdim=True))
            if self.use_reliability_gate:
                gate_logits = self.reliability_gates[i](torch.cat([c, detail[i], qi], dim=1))
                reliability = torch.sigmoid(gate_logits)
                reliability_maps.append(reliability)
                qualified_detail.append(detail[i] + reliability * f if self.residual_refinement else detail[i] * (2.0 * reliability))
        out: dict[str, list[Tensor] | Tensor] = {"descriptor": q, "features": feats, "maps": maps}
        if self.use_reliability_gate:
            out["reliability_maps"] = reliability_maps
            out["qualified_detail"] = qualified_detail
        return out


class ProposalNet(nn.Module):
    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.calign = nn.Sequential(nn.Conv2d(c, n, 1, bias=False), LayerNorm2d(n))
        self.net = nn.Sequential(nn.Conv2d(n * 3, n, 3, padding=1, bias=False), LayerNorm2d(n), nn.GELU(), ConvNeXtBlock(n), nn.Conv2d(n, 1, 1))
    def forward(self, c: Tensor, n: Tensor) -> Tensor:
        ca = self.calign(c)
        if ca.shape[-2:] != n.shape[-2:]: ca = F.interpolate(ca, size=n.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(self.net(torch.cat([n, ca, (norm2d(n)-norm2d(ca)).abs()], 1)))


class ReferenceNet(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        h = max(16, ch // 2)
        self.mix = nn.Sequential(nn.Conv2d(ch + 1, h, 1, bias=False), LayerNorm2d(h), nn.GELU(), nn.Conv2d(h, 1, 1))
        self.strength = nn.Sequential(nn.Conv2d(ch + 1, h, 1, bias=False), LayerNorm2d(h), nn.GELU(), nn.Conv2d(h, 1, 1))
    def forward(self, n: Tensor, m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b = (1 - m).detach()
        den = F.avg_pool2d(b, 7, 1, 3, count_include_pad=False).clamp_min(1e-4)
        local = F.avg_pool2d(n * b, 7, 1, 3, count_include_pad=False) / den
        gden = b.sum((2, 3), keepdim=True).clamp_min(1.0)
        glob = ((n * b).sum((2, 3), keepdim=True) / gden).expand_as(n)
        x = torch.cat([n, m], 1)
        mix = torch.sigmoid(self.mix(x))
        h = torch.sigmoid(self.strength(x))
        ref = mix * local + (1 - mix) * glob
        ncf = n + m * h * (ref - n)
        energy = (m * h * (ref - n)).abs().mean(1, keepdim=True)
        return ncf, ref, energy


class VerifierNet(nn.Module):
    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.calign = nn.Sequential(nn.Conv2d(c, n, 1, bias=False), LayerNorm2d(n))
        self.net = nn.Sequential(nn.Conv2d(n * 3, n, 3, padding=1, bias=False), LayerNorm2d(n), nn.GELU(), ConvNeXtBlock(n), nn.Conv2d(n, 1, 1))
    def forward(self, c: Tensor, n: Tensor) -> Tensor:
        ca = self.calign(c)
        if ca.shape[-2:] != n.shape[-2:]: ca = F.interpolate(ca, size=n.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([n, ca, (norm2d(n)-norm2d(ca)).abs()], 1))


class HCV2(nn.Module):
    def __init__(self, content_channels=(256,512,1024), detail_channels=(64,128,256), mandatory_q_fusion: bool = False, residual_verification: bool = False) -> None:
        super().__init__()
        self.mandatory_q_fusion = bool(mandatory_q_fusion)
        self.residual_verification = bool(residual_verification)
        self.q_alpha = nn.Parameter(torch.zeros(3))
        self.q_fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(n * 2, n, 1, bias=False), LayerNorm2d(n), nn.GELU())
            for n in detail_channels
        ]) if self.mandatory_q_fusion else nn.ModuleList()
        self.lambda_1 = nn.Parameter(torch.tensor(-10.0))
        self.lambda_2 = nn.Parameter(torch.tensor(-10.0))
        self.proposals = nn.ModuleList([ProposalNet(c,n) for c,n in zip(content_channels, detail_channels)])
        self.refs = nn.ModuleList([ReferenceNet(n) for n in detail_channels])
        self.verifiers = nn.ModuleList([VerifierNet(c,n) for c,n in zip(content_channels, detail_channels)])
        self.verified_heads = nn.ModuleList([nn.Sequential(
            nn.Conv2d(n + 3, n, 3, padding=1, bias=False), LayerNorm2d(n), nn.GELU(),
            ConvNeXtBlock(n), nn.Conv2d(n, n, 1)
        ) for n in detail_channels])
    def _apply_q(self, detail: list[Tensor], qtr: dict | None) -> list[Tensor]:
        qf = [] if qtr is None else qtr.get("features", [])
        out = []
        for i in range(3):
            q = torch.zeros_like(detail[i]) if len(qf) <= i else qf[i]
            if q.shape[-2:] != detail[i].shape[-2:]:
                q = F.interpolate(q, size=detail[i].shape[-2:], mode="bilinear", align_corners=False)
            if self.mandatory_q_fusion:
                out.append(self.q_fusions[i](torch.cat([detail[i], q], dim=1)))
            else:
                out.append(detail[i] + torch.tanh(self.q_alpha[i]) * q)
        return out
    def forward(self, content: list[Tensor], detail: list[Tensor], qtr: dict | None = None) -> dict[str, list[Tensor] | Tensor]:
        d = self._apply_q(detail, qtr)
        local = [self.proposals[i](content[i], d[i]) for i in range(3)]
        m3 = local[2]
        m2 = 1 - (1 - local[1]) * (1 - torch.sigmoid(self.lambda_2) * F.interpolate(m3, size=local[1].shape[-2:], mode="bilinear", align_corners=False))
        m1 = 1 - (1 - local[0]) * (1 - torch.sigmoid(self.lambda_1) * F.interpolate(m2, size=local[0].shape[-2:], mode="bilinear", align_corners=False))
        props = [m1.clamp(0,1), m2.clamp(0,1), m3.clamp(0,1)]
        ncf, refs, en, vf_logits, vcf_logits, pf, pcf, delta = [], [], [], [], [], [], [], []
        for i in range(3):
            cfi, ri, ei = self.refs[i](d[i], props[i])
            vfi = self.verifiers[i](content[i], d[i])
            vcfi = self.verifiers[i](content[i], cfi)
            pfi, pcfi = torch.sigmoid(vfi), torch.sigmoid(vcfi)
            ncf.append(cfi); refs.append(ri); en.append(ei); vf_logits.append(vfi); vcf_logits.append(vcfi); pf.append(pfi); pcf.append(pcfi); delta.append(pfi - pcfi)
        corrections = [self.verified_heads[i](torch.cat([d[i], props[i], delta[i], en[i]], dim=1)) for i in range(3)]
        verified = [d[i] + 0.1 * corrections[i] for i in range(3)] if self.residual_verification else corrections
        return {"local_proposals": local, "proposals": props, "counterfactuals": ncf, "references": refs, "energies": en, "v_factual_logits": vf_logits, "v_counterfactual_logits": vcf_logits, "p_factual": pf, "p_counterfactual": pcf, "deltas": delta, "verified_features": verified, "verification_corrections": corrections, "q_alpha": self.q_alpha, "lambda": torch.stack([self.lambda_1, self.lambda_2])}


class DHER(nn.Module):
    def __init__(self, detail_ch: int = 64, decoder_ch: int = 128) -> None:
        super().__init__()
        in_ch = decoder_ch + detail_ch * 2 + 5
        self.evidence = nn.Sequential(nn.Conv2d(in_ch, decoder_ch, 1, bias=False), LayerNorm2d(decoder_ch), nn.GELU(), ConvNeXtBlock(decoder_ch))
        self.support = nn.Conv2d(decoder_ch + 2, 1, 1)
        self.refute = nn.Conv2d(decoder_ch + 2, 1, 1)
        self.reexam = nn.Sequential(nn.Conv2d(decoder_ch + detail_ch * 2 + 4, decoder_ch, 3, padding=1, bias=False), LayerNorm2d(decoder_ch), nn.GELU(), ConvNeXtBlock(decoder_ch), nn.Conv2d(decoder_ch, decoder_ch, 1))
        self.gate = nn.Sequential(nn.Conv2d(2, decoder_ch, 1), nn.Sigmoid())
        self.alpha_re = nn.Parameter(torch.tensor(0.0))
    def forward(self, d1: Tensor, n1: Tensor, hcv: dict, qtr: dict | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        m = hcv["proposals"][0]; delta = hcv["deltas"][0]; energy = hcv["energies"][0]
        if n1.shape[-2:] != d1.shape[-2:]: n1 = F.interpolate(n1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        if qtr is not None:
            q1 = qtr["features"][0]
            if q1.shape[-2:] != d1.shape[-2:]: q1 = F.interpolate(q1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        else:
            q1 = torch.zeros_like(n1)
        vals = []
        for x in (m, delta, delta.abs(), energy):
            vals.append(F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != d1.shape[-2:] else x)
        m, delta, absd, energy = vals
        e = self.evidence(torch.cat([d1, n1, q1, m, delta, absd, F.relu(delta), F.relu(-delta)], 1))
        s = F.softplus(self.support(torch.cat([e, F.relu(delta), absd], 1)))
        r = F.softplus(self.refute(torch.cat([e, F.relu(-delta), absd], 1)))
        mass = s + r
        kappa = (2 * torch.minimum(s, r) / (mass + 1e-6)) * (1 - torch.exp(-mass))
        re = self.reexam(torch.cat([d1, n1, q1, delta, absd, s, r], 1))
        g = self.gate(torch.cat([kappa, mass], 1))
        write = torch.tanh(self.alpha_re) * g * re
        return d1 + write, {"support": s, "refutation": r, "conflict": kappa, "mass": mass, "write": write, "alpha_re": self.alpha_re}


class ScaleEvidenceHead(nn.Module):
    def __init__(self, detail_ch: int) -> None:
        super().__init__()
        hidden = max(32, detail_ch // 2)
        self.support = nn.Sequential(
            nn.Conv2d(detail_ch + 3, hidden, 3, padding=1, bias=False), LayerNorm2d(hidden), nn.GELU(),
            ConvNeXtBlock(hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        self.refutation = nn.Sequential(
            nn.Conv2d(detail_ch + 3, hidden, 3, padding=1, bias=False), LayerNorm2d(hidden), nn.GELU(),
            ConvNeXtBlock(hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, n: Tensor, m: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        if m.shape[-2:] != n.shape[-2:]:
            m = F.interpolate(m, size=n.shape[-2:], mode="bilinear", align_corners=False)
        if delta.shape[-2:] != n.shape[-2:]:
            delta = F.interpolate(delta, size=n.shape[-2:], mode="bilinear", align_corners=False)
        delta_pos = F.relu(delta)
        delta_neg = F.relu(-delta)
        abs_delta = delta.abs()
        support = F.softplus(self.support(torch.cat([n, m, delta_pos, abs_delta], dim=1)))
        refutation = F.softplus(self.refutation(torch.cat([n, 1 - m, delta_neg, abs_delta], dim=1)))
        return support, refutation


class DHERv2(nn.Module):
    """Asymmetric evidence routing with verified support shortcut.

    QTR/HCV remain evidence generators. This consumer only reads s1/s2 evidence and
    writes feature-space residuals through zero-initialized scalar gates.
    """

    def __init__(self, detail_channels=(64, 128), decoder_ch: int = 128) -> None:
        super().__init__()
        self.scale_heads = nn.ModuleList([ScaleEvidenceHead(c) for c in detail_channels[:2]])
        self.support_fuse = nn.Sequential(
            nn.Conv2d(6, decoder_ch, 3, padding=1, bias=False), LayerNorm2d(decoder_ch), nn.GELU(),
            ConvNeXtBlock(decoder_ch),
            nn.Conv2d(decoder_ch, decoder_ch, 1),
        )
        self.support_gate = nn.Sequential(nn.Conv2d(3, decoder_ch, 1), nn.Sigmoid())
        self.reexam = nn.Sequential(
            nn.Conv2d(decoder_ch + detail_channels[0] + 8, decoder_ch, 3, padding=1, bias=False),
            LayerNorm2d(decoder_ch), nn.GELU(), ConvNeXtBlock(decoder_ch), nn.Conv2d(decoder_ch, decoder_ch, 1),
        )
        nn.init.zeros_(self.support_fuse[-1].weight)
        nn.init.zeros_(self.support_fuse[-1].bias)
        nn.init.zeros_(self.reexam[-1].weight)
        nn.init.zeros_(self.reexam[-1].bias)
        self.alpha_support = nn.Parameter(torch.tensor(0.25))
        self.alpha_conflict = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _at(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    @staticmethod
    def _conflict(s: Tensor, r: Tensor) -> Tensor:
        mass = s + r
        return (2 * torch.minimum(s, r) / (mass + 1e-6)) * (1 - torch.exp(-mass))

    def forward(self, d1: Tensor, detail: list[Tensor], hcv: dict, qtr: dict | None = None) -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        _ = qtr
        supports, refutations, conflicts, delta_pos, props, deltas = [], [], [], [], [], []
        for i in range(2):
            n = detail[i]
            m = hcv["proposals"][i]
            delta = hcv["deltas"][i]
            s, r = self.scale_heads[i](n, m, delta)
            supports.append(s)
            refutations.append(r)
            conflicts.append(self._conflict(s, r))
            props.append(self._at(m, s))
            deltas.append(self._at(delta, s))
            delta_pos.append(F.relu(self._at(delta, s)))
        n1 = self._at(detail[0], d1)
        s1 = self._at(supports[0], d1)
        r1 = self._at(refutations[0], d1)
        k1 = self._at(conflicts[0], d1)
        dlt1 = self._at(deltas[0], d1)
        dp1 = F.relu(dlt1)
        m1 = self._at(props[0], d1)
        s2 = self._at(supports[1], d1)
        r2 = self._at(refutations[1], d1)
        k2 = self._at(conflicts[1], d1)
        dlt2 = self._at(deltas[1], d1)
        dp2 = F.relu(dlt2)
        m2 = self._at(props[1], d1)
        e_support = self.support_fuse(torch.cat([s1, dp1, m1, s2, dp2, m2], dim=1))
        g_support = self.support_gate(torch.cat([s1, dp1, m1], dim=1))
        f_re = self.reexam(torch.cat([d1, n1, s1, r1, k1, dlt1, s2, r2, k2, dlt2], dim=1))
        g_conflict = (k1 + k2).clamp(0, 1)
        support_write = torch.tanh(self.alpha_support) * g_support * e_support
        conflict_write = torch.tanh(self.alpha_conflict) * g_conflict * f_re
        out = d1 + support_write + conflict_write
        return out, {
            "support": supports,
            "refutation": refutations,
            "conflict": conflicts,
            "support_gate": g_support,
            "conflict_gate": g_conflict,
            "support_write": support_write,
            "conflict_write": conflict_write,
            "alpha_support": self.alpha_support,
            "alpha_conflict": self.alpha_conflict,
            "d1_rms": d1.detach().float().square().mean().sqrt().reshape(1),
            "support_write_rms": support_write.detach().float().square().mean().sqrt().reshape(1),
            "conflict_write_rms": conflict_write.detach().float().square().mean().sqrt().reshape(1),
        }


class DHERv21(nn.Module):
    """Sparse conflict routing plus missed-support routing for DHER-v2.1."""

    def __init__(self, detail_channels=(64, 128), decoder_ch: int = 128) -> None:
        super().__init__()
        self.scale_heads = nn.ModuleList([ScaleEvidenceHead(c) for c in detail_channels[:2]])
        self.support_fuse = nn.Sequential(
            nn.Conv2d(6, decoder_ch, 3, padding=1, bias=False), LayerNorm2d(decoder_ch), nn.GELU(),
            ConvNeXtBlock(decoder_ch), nn.Conv2d(decoder_ch, decoder_ch, 1),
        )
        self.reexam = nn.Sequential(
            nn.Conv2d(decoder_ch + detail_channels[0] + 8, decoder_ch, 3, padding=1, bias=False),
            LayerNorm2d(decoder_ch), nn.GELU(), ConvNeXtBlock(decoder_ch), nn.Conv2d(decoder_ch, decoder_ch, 1),
        )
        nn.init.zeros_(self.support_fuse[-1].weight)
        nn.init.zeros_(self.support_fuse[-1].bias)
        nn.init.zeros_(self.reexam[-1].weight)
        nn.init.zeros_(self.reexam[-1].bias)
        self.alpha_support = nn.Parameter(torch.tensor(0.25))
        self.alpha_conflict = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _at(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    @staticmethod
    def _conflict(s: Tensor, r: Tensor) -> Tensor:
        mass = s + r
        return (2 * torch.minimum(s, r) / (mass + 1e-6)) * (1 - torch.exp(-mass))

    @staticmethod
    def _spatial_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
        m = x.mean((2, 3), keepdim=True)
        sd = x.std((2, 3), keepdim=True, unbiased=False)
        return (x - m) / (sd + eps)

    def forward(self, d1: Tensor, detail: list[Tensor], hcv: dict, qtr: dict | None = None, base_logits_d1: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        _ = qtr
        supports, refutations, conflicts, props, deltas = [], [], [], [], []
        for i in range(2):
            n = detail[i]
            m = hcv["proposals"][i]
            delta = hcv["deltas"][i]
            s, r = self.scale_heads[i](n, m, delta)
            supports.append(s)
            refutations.append(r)
            conflicts.append(self._conflict(s, r))
            props.append(self._at(m, s))
            deltas.append(self._at(delta, s))
        n1 = self._at(detail[0], d1)
        s1 = self._at(supports[0], d1)
        r1 = self._at(refutations[0], d1)
        k1 = self._at(conflicts[0], d1)
        dlt1 = self._at(deltas[0], d1)
        dp1 = F.relu(dlt1)
        m1 = self._at(props[0], d1)
        s2 = self._at(supports[1], d1)
        r2 = self._at(refutations[1], d1)
        k2 = self._at(conflicts[1], d1)
        dlt2 = self._at(deltas[1], d1)
        dp2 = F.relu(dlt2)
        m2 = self._at(props[1], d1)

        if base_logits_d1 is None:
            base_logits_d1 = d1.new_zeros((d1.shape[0], 1, d1.shape[2], d1.shape[3]))
        if base_logits_d1.shape[-2:] != d1.shape[-2:]:
            base_logits_d1 = F.interpolate(base_logits_d1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        p_base = base_logits_d1.sigmoid().detach()
        uncertainty = 4 * p_base * (1 - p_base)

        margin = self._spatial_norm(s1 - r1) + 0.5 * self._spatial_norm(s2 - r2)
        support_confidence = torch.sigmoid(margin)
        miss_probability = 1 - p_base
        proposal_prior = (m1 + 0.5 * m2).clamp(0, 1)
        g_support = support_confidence * miss_probability * proposal_prior

        kappa_fused = 0.5 * (k1 + k2)
        g_conflict = kappa_fused * uncertainty

        e_support = self.support_fuse(torch.cat([s1, dp1, m1, s2, dp2, m2], dim=1))
        f_re = self.reexam(torch.cat([d1, n1, s1, r1, k1, dlt1, s2, r2, k2, dlt2], dim=1))
        support_write = torch.tanh(self.alpha_support) * g_support * e_support
        conflict_write = torch.tanh(self.alpha_conflict) * g_conflict * f_re
        return d1 + support_write + conflict_write, {
            "support": supports,
            "refutation": refutations,
            "conflict": conflicts,
            "support_gate": g_support,
            "conflict_gate": g_conflict,
            "support_write": support_write,
            "conflict_write": conflict_write,
            "alpha_support": self.alpha_support,
            "alpha_conflict": self.alpha_conflict,
            "d1_rms": d1.detach().float().square().mean().sqrt().reshape(1),
            "support_write_rms": support_write.detach().float().square().mean().sqrt().reshape(1),
            "conflict_write_rms": conflict_write.detach().float().square().mean().sqrt().reshape(1),
        }


class SameImageCorrespondenceBooster(nn.Module):
    """Non-local same-image correspondence cue injected at the 1/8 decoder level."""

    def __init__(self, content_ch: int = 512, proj_ch: int = 64, decoder_ch: int = 128, topk: int = 4, local_radius: int = 4, chunk_size: int = 512) -> None:
        super().__init__()
        self.topk = int(topk)
        self.local_radius = int(local_radius)
        self.chunk_size = int(chunk_size)
        self.project = nn.Sequential(nn.Conv2d(content_ch, proj_ch, 1, bias=False), LayerNorm2d(proj_ch), nn.GELU())
        rep_ch = 2 + proj_ch + 2
        self.adapter = nn.Sequential(
            nn.Conv2d(rep_ch, decoder_ch, 1, bias=False),
            LayerNorm2d(decoder_ch),
            nn.GELU(),
            ConvNeXtBlock(decoder_ch),
            nn.Conv2d(decoder_ch, decoder_ch, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    @staticmethod
    def _coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        denom_y = max(1, h - 1)
        denom_x = max(1, w - 1)
        return torch.stack([yy.reshape(-1).to(dtype) / denom_y, xx.reshape(-1).to(dtype) / denom_x], dim=1)

    def _nonlocal_topk(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, c, h, w = feat.shape
        n = h * w
        k = min(self.topk, max(1, n - 1))
        flat = feat.flatten(2).transpose(1, 2).contiguous()
        coords = self._coords(h, w, feat.device, feat.dtype)
        all_idx = torch.arange(n, device=feat.device)
        max_vals, mean_vals, best_idx = [], [], []
        radius = self.local_radius
        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            q = flat[:, start:end, :]
            sim = torch.bmm(q, flat.transpose(1, 2))
            q_idx = all_idx[start:end]
            qy = torch.div(q_idx, w, rounding_mode="floor")[:, None]
            qx = (q_idx % w)[:, None]
            ay = torch.div(all_idx, w, rounding_mode="floor")[None, :]
            ax = (all_idx % w)[None, :]
            local = (qy - ay).abs().le(radius) & (qx - ax).abs().le(radius)
            sim = sim.masked_fill(local[None, :, :], -2.0)
            vals, idx = torch.topk(sim, k=k, dim=-1)
            max_vals.append(vals[:, :, :1])
            mean_vals.append(vals.mean(dim=-1, keepdim=True))
            best_idx.append(idx[:, :, 0])
        max_sim = torch.cat(max_vals, dim=1).transpose(1, 2).reshape(b, 1, h, w)
        mean_sim = torch.cat(mean_vals, dim=1).transpose(1, 2).reshape(b, 1, h, w)
        best = torch.cat(best_idx, dim=1)
        gather = best[:, :, None].expand(-1, -1, c)
        best_feat = torch.gather(flat, 1, gather)
        diff = (flat - best_feat).abs().transpose(1, 2).reshape(b, c, h, w)
        best_coords = coords[best.reshape(-1)].reshape(b, n, 2)
        delta = best_coords - coords[None, :, :]
        dy = delta[:, :, 0].reshape(b, 1, h, w)
        dx = delta[:, :, 1].reshape(b, 1, h, w)
        return max_sim, mean_sim, diff, dx, dy

    def forward(self, c2: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        projected = self.project(c2)
        feat = F.normalize(projected, p=2, dim=1, eps=1e-6)
        max_sim, mean_sim, diff, dx, dy = self._nonlocal_topk(feat)
        rep = torch.cat([max_sim, mean_sim, diff, dx, dy], dim=1)
        write = self.adapter(rep)
        return write, {
            "write": write,
            "max_similarity": max_sim,
            "mean_topk_similarity": mean_sim,
            "diff_abs_mean": diff.mean(1, keepdim=True),
            "dx": dx,
            "dy": dy,
        }



class ContentQTRContextBlock(nn.Module):
    """Zero-effect multi-dilation context used only before QTR."""

    def __init__(self, channels: tuple[int, int, int] = (256, 512, 1024)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for c in channels:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, dilation=1, groups=c, bias=False), LayerNorm2d(c), nn.GELU(),
                nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False), LayerNorm2d(c), nn.GELU(),
                nn.Conv2d(c, c, 3, padding=3, dilation=3, groups=c, bias=False), LayerNorm2d(c), nn.GELU(),
                nn.Conv2d(c, c, 1),
            ))
        for block in self.blocks:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(self, content: list[Tensor]) -> list[Tensor]:
        out = list(content)
        for i, block in enumerate(self.blocks):
            out[i] = content[i] + block(content[i])
        return out


class ResidualMaskRefinement(nn.Module):
    """Zero-effect residual correction of final mask logits."""

    def __init__(self, decoder_channels: int = 128) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, d1: Tensor, logits_low: Tensor) -> tuple[Tensor, Tensor]:
        delta = self.delta(d1)
        return logits_low + torch.tanh(self.alpha) * delta, delta


class RegionRefineHead(nn.Module):
    """Zero-initialized local reconstruction head for mask-logit residuals."""

    def __init__(self, decoder_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(decoder_channels + 5, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _at(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    def forward(self, d1: Tensor, logits_low: Tensor, hcv: dict | None = None) -> tuple[Tensor, Tensor]:
        if hcv is None:
            zero = logits_low.new_zeros(logits_low.shape)
            cues = [zero, zero, zero, zero]
        else:
            cues = [
                self._at(hcv["proposals"][0], logits_low),
                self._at(hcv["proposals"][1], logits_low),
                self._at(hcv["deltas"][0], logits_low),
                self._at(hcv["deltas"][1], logits_low),
            ]
        residual = self.net(torch.cat([d1, *cues, logits_low], dim=1))
        return logits_low + residual, residual


class ForensicP2Preservation(nn.Module):
    """Stride-4 forensic bypass fused into the final high-resolution decoder feature."""

    def __init__(self, p2_channels: int = 64, decoder_channels: int = 128, proj_channels: int = 64) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(p2_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels + proj_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
        )
        self._init_identity_fusion(decoder_channels, proj_channels)

    def _init_identity_fusion(self, decoder_channels: int, proj_channels: int) -> None:
        conv = self.fuse[0]
        bn = self.fuse[1]
        nn.init.zeros_(conv.weight)
        with torch.no_grad():
            for c in range(decoder_channels):
                conv.weight[c, c, 1, 1] = 1.0
        nn.init.constant_(bn.weight, (1.0 + bn.eps) ** 0.5)
        nn.init.zeros_(bn.bias)
        nn.init.zeros_(bn.running_mean)
        nn.init.ones_(bn.running_var)

    def forward(self, d1: Tensor, p2: Tensor) -> Tensor:
        p = self.project(p2)
        if p.shape[-2:] != d1.shape[-2:]:
            p = F.interpolate(p, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        bn = self.fuse[1]
        with torch.backends.cudnn.flags(allow_tf32=False):
            y = self.fuse[0](torch.cat([d1, p], dim=1))
        return F.batch_norm(y, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False, eps=bn.eps)

class E15EvidenceAdapter(nn.Module):
    """Lightweight P/Delta/Q cue adapter with a zero-start residual gate."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.beta = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _cue(x: Tensor | None, ref: Tensor) -> Tensor:
        if x is None:
            return ref.new_zeros((ref.shape[0], 1, ref.shape[2], ref.shape[3]))
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    def forward(self, ref: Tensor, proposal: Tensor | None, delta: Tensor | None, qfeat: Tensor | None) -> Tensor:
        return torch.tanh(self.beta) * self.net(torch.cat([self._cue(proposal, ref), self._cue(delta, ref), self._cue(qfeat, ref)], dim=1))


class E15DecoderStage(nn.Module):
    def __init__(self, content_channels: int, detail_channels: int, out_channels: int = 128, has_parent: bool = True) -> None:
        super().__init__()
        self.content = nn.Sequential(nn.Conv2d(content_channels, out_channels, 1, bias=False), LayerNorm2d(out_channels))
        self.detail = nn.Sequential(nn.Conv2d(detail_channels, out_channels, 1, bias=False), LayerNorm2d(out_channels))
        self.has_parent = bool(has_parent)
        in_channels = out_channels * (3 if self.has_parent else 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            ConvNeXtBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, content: Tensor, detail: Tensor, parent: Tensor | None = None) -> Tensor:
        c = self.content(content)
        n = self.detail(detail)
        parts = [c, n]
        if self.has_parent:
            if parent is None:
                raise ValueError("parent feature is required")
            parts.append(F.interpolate(parent, size=c.shape[-2:], mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(parts, dim=1))


class E15HighResBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class E15SpatialStem(nn.Module):
    def __init__(self, full_channels: int = 32, half_channels: int = 64) -> None:
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(3, full_channels, 3, padding=1, bias=False),
            LayerNorm2d(full_channels),
            nn.GELU(),
            nn.Conv2d(full_channels, full_channels, 3, padding=1, bias=False),
            LayerNorm2d(full_channels),
            nn.GELU(),
        )
        self.s2 = nn.Sequential(
            nn.Conv2d(full_channels, half_channels, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(half_channels),
            nn.GELU(),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        s1 = self.s1(image)
        s2 = self.s2(s1)
        return s1, s2


class E15HighResolutionDecoder(nn.Module):
    """Coarse-to-fine evidence reconstruction plus learned high-resolution mask residual."""

    def __init__(
        self,
        content_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        detail_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.stage4 = E15DecoderStage(content_channels[3], detail_channels[3], decoder_channels, has_parent=False)
        self.stage3 = E15DecoderStage(content_channels[2], detail_channels[2], decoder_channels, has_parent=True)
        self.stage2 = E15DecoderStage(content_channels[1], detail_channels[1], decoder_channels, has_parent=True)
        self.stage1 = E15DecoderStage(content_channels[0], detail_channels[0], decoder_channels, has_parent=True)
        self.evidence_adapters = nn.ModuleList(E15EvidenceAdapter(decoder_channels) for _ in range(4))
        self.spatial_stem = E15SpatialStem()
        self.up_to_half = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.r2_fuse = nn.Sequential(nn.Conv2d(decoder_channels + 64 + 2, decoder_channels, 3, padding=1, bias=False), LayerNorm2d(decoder_channels), nn.GELU())
        self.r2_block = E15HighResBlock(decoder_channels)
        self.up_to_full = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.r1_fuse = nn.Sequential(nn.Conv2d(decoder_channels + 32 + 2, decoder_channels, 3, padding=1, bias=False), LayerNorm2d(decoder_channels), nn.GELU())
        self.r1_block = E15HighResBlock(decoder_channels)
        self.region_head = nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False), LayerNorm2d(decoder_channels), nn.GELU(), nn.Conv2d(decoder_channels, 1, 1))
        self.boundary_head = nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False), LayerNorm2d(decoder_channels // 2), nn.GELU(), nn.Conv2d(decoder_channels // 2, 1, 1))
        self.aux_d1 = nn.Conv2d(decoder_channels, 1, 1)
        self.aux_d2 = nn.Conv2d(decoder_channels, 1, 1)
        self.aux_d3 = nn.Conv2d(decoder_channels, 1, 1)
        nn.init.zeros_(self.region_head[-1].weight)
        nn.init.zeros_(self.region_head[-1].bias)

    @staticmethod
    def _hcv_pair(hcv: dict | None, index: int) -> tuple[Tensor | None, Tensor | None]:
        if hcv is None or index >= 3:
            return None, None
        return hcv["proposals"][index], hcv["deltas"][index]

    @staticmethod
    def _qfeat(qtr: dict | None, index: int) -> Tensor | None:
        if qtr is None:
            return None
        qf = qtr.get("features", [])
        return qf[index] if index < len(qf) else None

    @staticmethod
    def _cue_at(x: Tensor | None, ref: Tensor) -> Tensor:
        if x is None:
            return ref.new_zeros((ref.shape[0], 1, ref.shape[2], ref.shape[3]))
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    def coarse(self, content: list[Tensor], detail: list[Tensor], qtr: dict | None, hcv: dict | None) -> list[Tensor]:
        p3, dlt3 = self._hcv_pair(hcv, 2)
        d4 = self.stage4(content[3], detail[3])
        d4 = d4 + self.evidence_adapters[3](d4, p3, dlt3, self._qfeat(qtr, 2))
        p2, dlt2 = self._hcv_pair(hcv, 2)
        d3 = self.stage3(content[2], detail[2], d4)
        d3 = d3 + self.evidence_adapters[2](d3, p2, dlt2, self._qfeat(qtr, 2))
        p1, dlt1 = self._hcv_pair(hcv, 1)
        d2 = self.stage2(content[1], detail[1], d3)
        d2 = d2 + self.evidence_adapters[1](d2, p1, dlt1, self._qfeat(qtr, 1))
        p0, dlt0 = self._hcv_pair(hcv, 0)
        d1 = self.stage1(content[0], detail[0], d2)
        d1 = d1 + self.evidence_adapters[0](d1, p0, dlt0, self._qfeat(qtr, 0))
        return [d1, d2, d3, d4]

    def reconstruct(self, image: Tensor, d1: Tensor, hcv: dict | None) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        s1, s2 = self.spatial_stem(image)
        p1 = hcv["proposals"][0] if hcv is not None else None
        dlt1 = hcv["deltas"][0] if hcv is not None else None
        half = self.up_to_half(d1)
        if half.shape[-2:] != s2.shape[-2:]:
            half = F.interpolate(half, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        r2 = self.r2_block(self.r2_fuse(torch.cat([half, s2, self._cue_at(p1, s2), self._cue_at(dlt1, s2)], dim=1)))
        full = self.up_to_full(r2)
        if full.shape[-2:] != s1.shape[-2:]:
            full = F.interpolate(full, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        r1 = self.r1_block(self.r1_fuse(torch.cat([full, s1, self._cue_at(p1, s1), self._cue_at(dlt1, s1)], dim=1)))
        residual = self.region_head(r1)
        boundary_logits = self.boundary_head(r1)
        return residual, boundary_logits, {"d1_input": d1, "r2": r2, "r1": r1}


class DCTEvidenceInjectionAdapter(nn.Module):
    def __init__(self, channels: int, zero_init: bool = True, use_gate: bool = False, gate_bias: float = -4.0) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(channels, 1, 3, padding=1) if use_gate else None
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, detail: Tensor, dct_feature: Tensor, available: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if dct_feature.shape[-2:] != detail.shape[-2:]:
            dct_feature = F.interpolate(dct_feature, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        if available.ndim == 1:
            available = available[:, None, None, None]
        elif available.ndim == 2:
            available = available[:, :, None, None]
        gate = torch.ones((detail.shape[0], 1, detail.shape[-2], detail.shape[-1]), dtype=detail.dtype, device=detail.device)
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(dct_feature))
        write = self.proj(dct_feature) * gate * available.to(dtype=detail.dtype, device=detail.device)
        return detail + write, write, gate


class SplicingRegionAffinity(nn.Module):
    """Learn donor/host affinity and use it for edge-aware decoder propagation."""

    def __init__(self, forensic_channels: int, dct_channels: int, decoder_channels: int, embedding_channels: int = 32) -> None:
        super().__init__()
        self.forensic_proj = nn.Conv2d(forensic_channels, embedding_channels, 1)
        self.dct_proj = nn.Conv2d(dct_channels, embedding_channels, 1)
        self.embedding = nn.Sequential(
            nn.Conv2d(embedding_channels * 2, embedding_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, embedding_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(embedding_channels, embedding_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, embedding_channels),
            nn.SiLU(inplace=True),
        )
        self.pair_score = nn.Sequential(
            nn.Conv2d(embedding_channels, embedding_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(embedding_channels // 2, 1, 1),
        )
        self.decoder_write = nn.Conv2d(decoder_channels, decoder_channels, 1)
        nn.init.zeros_(self.decoder_write.weight)
        nn.init.zeros_(self.decoder_write.bias)

    @staticmethod
    def _availability(available: Tensor, reference: Tensor) -> Tensor:
        if available.ndim == 1:
            available = available[:, None, None, None]
        elif available.ndim == 2:
            available = available[:, :, None, None]
        return available.to(dtype=reference.dtype, device=reference.device)

    def forward(
        self,
        d1: Tensor,
        forensic: Tensor,
        dct_feature: Tensor,
        dct_available: Tensor,
        disabled: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        size = d1.shape[-2:]
        forensic = F.interpolate(forensic, size=size, mode="bilinear", align_corners=False)
        dct_feature = F.interpolate(dct_feature, size=size, mode="bilinear", align_corners=False)
        available = self._availability(dct_available, d1)
        embedding = self.embedding(torch.cat([
            self.forensic_proj(forensic),
            self.dct_proj(dct_feature) * available,
        ], dim=1))

        horizontal_logits = self.pair_score((embedding[:, :, :, 1:] - embedding[:, :, :, :-1]).abs())
        vertical_logits = self.pair_score((embedding[:, :, 1:, :] - embedding[:, :, :-1, :]).abs())
        horizontal = horizontal_logits.sigmoid()
        vertical = vertical_logits.sigmoid()

        neighbour_sum = d1
        neighbour_sum = neighbour_sum + F.pad(horizontal * d1[:, :, :, 1:], (0, 1, 0, 0))
        neighbour_sum = neighbour_sum + F.pad(horizontal * d1[:, :, :, :-1], (1, 0, 0, 0))
        neighbour_sum = neighbour_sum + F.pad(vertical * d1[:, :, 1:, :], (0, 0, 0, 1))
        neighbour_sum = neighbour_sum + F.pad(vertical * d1[:, :, :-1, :], (0, 0, 1, 0))
        normalizer = d1.new_ones((d1.shape[0], 1, *size))
        normalizer = normalizer + F.pad(horizontal, (0, 1, 0, 0)) + F.pad(horizontal, (1, 0, 0, 0))
        normalizer = normalizer + F.pad(vertical, (0, 0, 0, 1)) + F.pad(vertical, (0, 0, 1, 0))
        propagated = neighbour_sum / normalizer.clamp_min(1.0)
        write = self.decoder_write(propagated - d1)
        if disabled:
            write = write * 0.0
        refined = d1 + write
        return refined, {
            "horizontal_logits": horizontal_logits,
            "vertical_logits": vertical_logits,
            "horizontal_affinity_mean": horizontal.detach().mean(),
            "vertical_affinity_mean": vertical.detach().mean(),
            "embedding_rms": embedding.detach().float().square().mean().sqrt(),
            "write_rms": write.detach().float().square().mean().sqrt(),
            "write_weight_rms": self.decoder_write.weight.detach().float().square().mean().sqrt(),
            "disabled": d1.new_tensor(float(disabled)),
        }


class DHEREvidenceVisibility(nn.Module):
    """Zero-init logit residual from existing HCV/DHER evidence maps."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(14, hidden_channels, 3, padding=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _at(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    def _terms(self, hcv: dict, dher: dict, index: int, ref: Tensor) -> list[Tensor]:
        proposal = self._at(hcv["proposals"][index], ref)
        delta = self._at(hcv["deltas"][index], ref)
        support = self._at(dher["support"][index], ref)
        conflict = self._at(dher["conflict"][index], ref)
        return [
            proposal,
            delta,
            delta.abs(),
            F.relu(delta),
            F.relu(-delta),
            torch.log1p(support.clamp_min(0)),
            conflict,
        ]

    def forward(self, logits_low: Tensor, hcv: dict, dher: dict) -> tuple[Tensor, dict[str, Tensor]]:
        evidence = torch.cat([*self._terms(hcv, dher, 0, logits_low), *self._terms(hcv, dher, 1, logits_low)], dim=1)
        residual = self.net(evidence)
        return logits_low + residual, {
            "residual": residual,
            "residual_rms": residual.detach().float().square().mean().sqrt().reshape(1),
            "residual_mean": residual.detach().float().mean().reshape(1),
            "input_rms": evidence.detach().float().square().mean().sqrt().reshape(1),
            "last_weight_rms": self.net[-1].weight.detach().float().square().mean().sqrt().reshape(1),
        }


class DHERCalibratedEvidenceWriteback(nn.Module):
    """Bounded gate-controlled evidence write-back on D1 and final logits.

    The global scales are zero-initialized, so adding this block preserves the
    source checkpoint prediction exactly at step0 while still letting the first
    backward pass decide whether evidence should become visible.
    """

    def __init__(
        self,
        decoder_channels: int = 128,
        hidden_channels: int = 32,
        gate_bias: float = -3.0,
        logit_limit: float = 0.75,
        feature_limit: float = 0.05,
    ) -> None:
        super().__init__()
        self.logit_limit = float(logit_limit)
        self.feature_limit = float(feature_limit)
        self.encoder = nn.Sequential(
            nn.Conv2d(14, hidden_channels, 3, padding=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.gate_head = nn.Conv2d(hidden_channels, 1, 1)
        self.logit_write = nn.Conv2d(hidden_channels, 1, 1)
        self.feature_write = nn.Conv2d(hidden_channels, decoder_channels, 1)
        self.logit_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.feature_scale_raw = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))

    @staticmethod
    def _at(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != ref.shape[-2:] else x

    def _terms(self, hcv: dict, dher: dict, index: int, ref: Tensor) -> list[Tensor]:
        proposal = self._at(hcv["proposals"][index], ref)
        delta = self._at(hcv["deltas"][index], ref)
        support = self._at(dher["support"][index], ref)
        conflict = self._at(dher["conflict"][index], ref)
        return [
            proposal,
            delta,
            delta.abs(),
            F.relu(delta),
            F.relu(-delta),
            torch.log1p(support.clamp_min(0)),
            conflict,
        ]

    def _encode(self, ref: Tensor, hcv: dict, dher: dict) -> tuple[Tensor, Tensor, Tensor]:
        evidence = torch.cat([*self._terms(hcv, dher, 0, ref), *self._terms(hcv, dher, 1, ref)], dim=1)
        hidden = self.encoder(evidence)
        gate = torch.sigmoid(self.gate_head(hidden))
        return evidence, hidden, gate

    def apply_feature(self, d1: Tensor, hcv: dict, dher: dict, disabled: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        evidence, hidden, gate = self._encode(d1, hcv, dher)
        scale = torch.tanh(self.feature_scale_raw) * self.feature_limit
        residual = scale * gate * torch.tanh(self.feature_write(hidden))
        if disabled:
            residual = residual * 0.0
        return d1 + residual, {
            "feature_residual": residual,
            "feature_residual_rms": residual.detach().float().square().mean().sqrt().reshape(1),
            "feature_scale": scale.detach().reshape(1),
            "gate_mean": gate.detach().float().mean().reshape(1),
            "gate_rms": gate.detach().float().square().mean().sqrt().reshape(1),
            "input_rms": evidence.detach().float().square().mean().sqrt().reshape(1),
        }

    def apply_logit(self, logits_low: Tensor, hcv: dict, dher: dict, disabled: bool = False) -> tuple[Tensor, dict[str, Tensor]]:
        _, hidden, gate = self._encode(logits_low, hcv, dher)
        scale = torch.tanh(self.logit_scale_raw) * self.logit_limit
        residual = scale * gate * torch.tanh(self.logit_write(hidden))
        if disabled:
            residual = residual * 0.0
        return logits_low + residual, {
            "logit_residual": residual,
            "logit_residual_rms": residual.detach().float().square().mean().sqrt().reshape(1),
            "logit_residual_mean": residual.detach().float().mean().reshape(1),
            "logit_scale": scale.detach().reshape(1),
            "logit_limit": logits_low.new_tensor(self.logit_limit).reshape(1),
            "feature_limit": logits_low.new_tensor(self.feature_limit).reshape(1),
        }


class FinalReTraceNet(nn.Module):
    def __init__(self, architecture: str = "retracenet_b0", decoder_channels: int = 128, residual_kernels: int = 8, forensic_channels=(64,128,256,512), pretrained_path: str | None = None, freeze_encoder_bn: bool = True, content_backbone: str = "resnet50", content_channels=(256,512,1024,2048), convnext_pretrained: bool = True, forensic_stream_type: str = "residual", dct_pretrained_path: str | None = None, freeze_forensic_bn: bool = False, p2_preserve_channels: int = 64, enable_mrfe: bool = False, enable_fft_prior: bool = False, fft_prior_cutoff: float = 0.18, enable_srm_prior: bool = False, enable_decoder_forensic_reconstruction: bool = False, decoder_forensic_reconstruction_proj_channels: int = 32, hrnet_model_name: str = "hrnet_w18", hrnet_pretrained: bool = True, enable_dct_parallel: bool = False, dct_injection_zero_init: bool = True, enable_dct_gate: bool = False, dct_gate_bias: float = -4.0, enable_dct_evidence_head: bool = False, enable_splicing_region_affinity: bool = False, splicing_affinity_channels: int = 32, enable_dher_visibility: bool = False, enable_dher_calibrated_writeback: bool = False, enable_effective_mechanism_chain: bool = False, enable_strict_serial_chain: bool = False, enable_balanced_residual_chain: bool = False) -> None:
        super().__init__()
        self.architecture = architecture
        self.content_backbone = content_backbone
        if content_backbone == "convnext_tiny":
            self.content_encoder = ConvNeXtTinyContentEncoder(pretrained_path=pretrained_path, pretrained=convnext_pretrained)
            content_channels = tuple(content_channels or ConvNeXtTinyContentEncoder.channels)
        elif content_backbone == "resnext101_32x8d":
            self.content_encoder = ResNeXt101_32x8dContentEncoder(pretrained_path=pretrained_path, freeze_bn=freeze_encoder_bn)
            content_channels = tuple(content_channels or ResNeXt101_32x8dContentEncoder.channels)
        elif content_backbone == "resnet101":
            self.content_encoder = ResNet101ContentEncoder(pretrained_path=pretrained_path, freeze_bn=freeze_encoder_bn)
            content_channels = tuple(content_channels or ResNet101ContentEncoder.channels)
        elif content_backbone == "resnet50":
            self.content_encoder = ResNet50ContentEncoder(pretrained_path=pretrained_path, freeze_bn=freeze_encoder_bn)
            content_channels = tuple(content_channels or ResNet50ContentEncoder.channels)
        elif content_backbone in {"hrnet_w18", "hrnet"}:
            self.content_encoder = HRNetW18ContentEncoder(
                model_name=hrnet_model_name,
                pretrained_path=pretrained_path,
                pretrained=hrnet_pretrained,
                out_channels=tuple(content_channels or HRNetW18ContentEncoder.channels),
            )
            content_channels = tuple(content_channels or HRNetW18ContentEncoder.channels)
        else:
            raise ValueError(f"unsupported content_backbone: {content_backbone}")
        self.forensic_stream_type = str(forensic_stream_type)

        if self.forensic_stream_type == "cue_fusion":
            if (enable_dct_parallel
                or not enable_balanced_residual_chain
                or enable_splicing_region_affinity):
                raise ValueError("Incompatible cue_fusion config")

            residual_encoder = LightweightForensicStream(
                residual_kernels=residual_kernels,
                channels=tuple(forensic_channels),
                enable_mrfe=enable_mrfe,
                enable_fft_prior=enable_fft_prior
                    or ("fft_prior" in architecture),
                fft_prior_cutoff=fft_prior_cutoff,
                enable_srm_prior=enable_srm_prior
                    or ("srm_prior" in architecture),
            )

            self.forensic_stream = UnifiedForensicStream(
                residual_encoder,
                DCTCueEncoder(
                    channels=tuple(forensic_channels),
                    pretrained_path=dct_pretrained_path,
                    freeze_bn=freeze_forensic_bn,
                ),
                nn.ModuleList([
                    BalancedForensicFusion(int(c))
                    for c in forensic_channels
                ]),
                nn.ModuleList([
                    DCTEvidenceInjectionAdapter(
                        int(c),
                        zero_init=bool(dct_injection_zero_init),
                        use_gate=bool(enable_dct_gate),
                        gate_bias=float(dct_gate_bias),
                    )
                    for c in forensic_channels
                ]),
                nn.Conv2d(
                    int(forensic_channels[0]), 1, 1
                ) if enable_dct_evidence_head else None,
            )

        elif self.forensic_stream_type in {"residual", "noise", "lightweight"}:

            self.forensic_stream = LightweightForensicStream(
                residual_kernels=residual_kernels,
                channels=tuple(forensic_channels),
                enable_mrfe=enable_mrfe,
                enable_fft_prior=enable_fft_prior or ("fft_prior" in architecture),
                fft_prior_cutoff=fft_prior_cutoff,
                enable_srm_prior=enable_srm_prior or ("srm_prior" in architecture),
            )
        elif self.forensic_stream_type in {"srm_guided", "srm_encoder", "forensic_plus"}:
            self.forensic_stream = SRMGuidedForensicStream(
                residual_kernels=residual_kernels,
                channels=tuple(forensic_channels),
                enable_mrfe=enable_mrfe or ("mrfe" in architecture),
            )
        elif self.forensic_stream_type in {"dct", "dct_compression", "dct_cue"}:
            self.forensic_stream = DCTCueEncoder(channels=tuple(forensic_channels), pretrained_path=dct_pretrained_path, freeze_bn=freeze_forensic_bn)
        elif self.forensic_stream_type in {"native_jpeg_hrnet", "native_jpeg_hrnet_w32"}:
            self.forensic_stream = NativeJPEGForensicHRNet(
                residual_kernels=residual_kernels,
                hr_channels=(32, 64, 128, 256),
                output_channels=tuple(forensic_channels),
            )
        else:
            raise ValueError(f"unsupported forensic_stream_type: {self.forensic_stream_type}")
        self.dct_branch = None
        self.dct_inject_adapters = None
        self.dct_evidence_head = None
        self.balanced_forensic_fusions = None
        self.disable_dct_evidence = False
        self.splicing_region_affinity = None
        self.disable_splicing_region = False
        self.dher_visibility = None
        self.disable_dher_visibility = False
        self.dher_calibrated_writeback = None
        self.disable_dher_calibrated_writeback = False
        # Explicit research ablations.  Modules remain registered so a full
        # checkpoint can be loaded strictly, but their forward paths are
        # bypassed without replacing learned outputs with trainable surrogates.
        self.ablate_qtr = False
        self.ablate_hcv = False
        self.ablate_dher = False
        if enable_dct_parallel:
            if isinstance(self.forensic_stream, DCTCueEncoder):
                raise ValueError("enable_dct_parallel keeps the existing forensic stream; use forensic_stream_type=residual/srm_guided")
            self.dct_branch = DCTCueEncoder(channels=tuple(forensic_channels), pretrained_path=dct_pretrained_path, freeze_bn=freeze_forensic_bn)
            self.dct_inject_adapters = nn.ModuleList(
                DCTEvidenceInjectionAdapter(int(ch), zero_init=bool(dct_injection_zero_init), use_gate=bool(enable_dct_gate), gate_bias=float(dct_gate_bias))
                for ch in tuple(forensic_channels)
            )
            if enable_balanced_residual_chain:
                self.balanced_forensic_fusions = nn.ModuleList(BalancedForensicFusion(int(ch)) for ch in tuple(forensic_channels))
            if enable_dct_evidence_head:
                self.dct_evidence_head = nn.Conv2d(int(forensic_channels[0]), 1, 1)
            if enable_splicing_region_affinity:
                self.splicing_region_affinity = SplicingRegionAffinity(
                    forensic_channels=int(forensic_channels[0]),
                    dct_channels=int(forensic_channels[0]),
                    decoder_channels=decoder_channels,
                    embedding_channels=int(splicing_affinity_channels),
                )
        self.decoder = ProgressiveDecoder(
            content_channels=tuple(content_channels),
            detail_channels=tuple(forensic_channels),
            decoder_channels=decoder_channels,
            enable_forensic_reconstruction=enable_decoder_forensic_reconstruction or ("decoder_mrfe_recon" in architecture),
            forensic_reconstruction_proj_channels=decoder_forensic_reconstruction_proj_channels,
        )
        self.content_context = ContentQTRContextBlock(channels=tuple(content_channels[:3])) if "context_qtr" in architecture else None
        self.mask_refine = ResidualMaskRefinement(decoder_channels=decoder_channels) if "mask_refine" in architecture else None
        self.region_refine = RegionRefineHead(decoder_channels=decoder_channels) if "region_refine" in architecture else None
        self.p2_preserve = ForensicP2Preservation(p2_channels=int(forensic_channels[0]), decoder_channels=decoder_channels, proj_channels=int(p2_preserve_channels)) if "p2" in architecture else None
        self.correspondence = SameImageCorrespondenceBooster(
            content_ch=int(content_channels[1]),
            proj_ch=64,
            decoder_ch=decoder_channels,
        ) if "correspondence" in architecture else None
        self.mask_head = nn.Conv2d(decoder_channels, 1, 1)
        self.log_var_head = nn.Conv2d(decoder_channels, 1, 1)
        self.aux_heads = nn.ModuleList(nn.Conv2d(decoder_channels, 1, 1) for _ in range(3))
        self.image_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(decoder_channels, 1))
        effective_chain = bool(enable_effective_mechanism_chain)
        self.qtr = QualityTraceQualification(use_reliability_gate=(effective_chain or enable_balanced_residual_chain or "qtr_gate" in architecture), residual_refinement=enable_balanced_residual_chain) if "qtr" in architecture else None
        self.hcv2 = HCV2(mandatory_q_fusion=effective_chain, residual_verification=enable_balanced_residual_chain) if ("hcv2" in architecture or "qtr" in architecture or "dher" in architecture) else None
        self.enable_strict_serial_chain = bool(enable_strict_serial_chain)
        self.enable_balanced_residual_chain = bool(enable_balanced_residual_chain)
        self.serial_coarse_adapter = nn.Sequential(nn.Conv2d(256, 512, 1, bias=False), LayerNorm2d(512), nn.GELU()) if self.enable_strict_serial_chain else None
        self.qtr_aux_heads = nn.ModuleList([nn.Conv2d(n, 1, 1) for n in forensic_channels[:3]]) if (self.enable_strict_serial_chain or self.enable_balanced_residual_chain) else nn.ModuleList()
        if "dher_v21" in architecture:
            self.dher = DHERv21(detail_channels=tuple(forensic_channels[:2]), decoder_ch=decoder_channels)
        elif "dher_v2" in architecture:
            self.dher = DHERv2(detail_channels=tuple(forensic_channels[:2]), decoder_ch=decoder_channels)
        elif "dher" in architecture:
            self.dher = DHER(decoder_ch=decoder_channels)
        else:
            self.dher = None
        if enable_balanced_residual_chain and isinstance(self.dher, (DHERv2, DHERv21)):
            with torch.no_grad():
                self.dher.alpha_support.fill_(0.1003353477)
                self.dher.alpha_conflict.fill_(0.1003353477)
            nn.init.normal_(self.dher.support_fuse[-1].weight, std=1e-3)
            nn.init.zeros_(self.dher.support_fuse[-1].bias)
            nn.init.normal_(self.dher.reexam[-1].weight, std=1e-3)
            nn.init.zeros_(self.dher.reexam[-1].bias)
        if enable_dher_visibility:
            if self.dher is None or self.hcv2 is None:
                raise ValueError("enable_dher_visibility requires HCV and DHER")
            self.dher_visibility = DHEREvidenceVisibility()
        if enable_dher_calibrated_writeback:
            if self.dher is None or self.hcv2 is None:
                raise ValueError("enable_dher_calibrated_writeback requires HCV and DHER")
            self.dher_calibrated_writeback = DHERCalibratedEvidenceWriteback(decoder_channels=decoder_channels)
        self.e15_decoder = E15HighResolutionDecoder(
            content_channels=tuple(content_channels),
            detail_channels=tuple(forensic_channels),
            decoder_channels=decoder_channels,
        ) if "e15_hr" in architecture else None
        self.e15_anchor_content_encoder = (
            ResNet50ContentEncoder(pretrained_path=None, freeze_bn=freeze_encoder_bn)
            if "e15_hr" in architecture
            else None
        )
    def _compat(self, logits, log_var, image_logit, aux, content):
        zeros = torch.zeros_like(logits)
        coarse = F.interpolate(logits, size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        strength = F.interpolate(logits.sigmoid().detach(), size=content[-1].shape[-2:], mode="bilinear", align_corners=False)
        out = {"mask_logits": logits, "base_logits": logits, "proposal_logits": logits, "coarse_logits": coarse, "factual_logits": logits, "log_var": log_var, "image_logit": image_logit, "aux_logits": aux, "content_features": content, "trace_features": content, "quality_descriptor": logits.new_zeros((logits.shape[0],8)), "scale_reliability": [logits.new_full((logits.shape[0],1,1,1),0.5) for _ in range(4)], "ledger_logits": [logits], "intervention_effects": [strength], "intervention_strengths": [strength]}
        out["counter"+"factual_logits"] = logits.detach(); out["ca"+"usal_effect"] = zeros; out["ledger_"+"con"+"flicts"] = [torch.zeros_like(strength)]
        return out

    def _apply_dher(self, d1: Tensor, decoded: list[Tensor], detail_for_model: list[Tensor], hcv: dict | None, qtr: dict | None) -> tuple[list[Tensor], dict | None]:
        dher_diag = None
        if self.dher is not None and hcv is not None:
            if isinstance(self.dher, DHERv21):
                base_logits_d1 = self.mask_head(d1).detach()
                d1, dher_diag = self.dher(d1, detail_for_model[:2], hcv, qtr, base_logits_d1)
            elif isinstance(self.dher, DHERv2):
                d1, dher_diag = self.dher(d1, detail_for_model[:2], hcv, qtr)
            else:
                d1, dher_diag = self.dher(d1, detail_for_model[0], hcv, qtr)
        return [d1, *decoded[1:]], dher_diag

    def _inject_dct_evidence(self, detail: list[Tensor], dct: Tensor | None, qtable: Tensor | None, dct_available: Tensor | None) -> tuple[list[Tensor], dict | None, list[Tensor] | None]:
        if self.dct_branch is None or self.dct_inject_adapters is None:
            return detail, None, None
        if dct is None or qtable is None:
            raise ValueError("enable_dct_parallel requires batch dct, qtable, and dct_available tensors")
        if dct_available is None:
            dct_available = torch.ones((dct.shape[0], 1), dtype=dct.dtype, device=dct.device)
        dct_features = self.dct_branch(dct, qtable)
        if bool(getattr(self, "disable_dct_evidence", False)):
            available = torch.zeros_like(dct_available, dtype=detail[0].dtype, device=detail[0].device)
        else:
            available = dct_available.to(dtype=detail[0].dtype, device=detail[0].device)
        fused, writes, gates = [], [], []
        for idx,(base, dct_feature, adapter) in enumerate(zip(detail, dct_features, self.dct_inject_adapters)):
            if self.balanced_forensic_fusions is not None:
                out,rnorm,dnorm=self.balanced_forensic_fusions[idx](base,dct_feature,available); write=out-base; gate=available[:,:,None,None]
            else:
                out, write, gate = adapter(base, dct_feature, available)
            fused.append(out); writes.append(write); gates.append(gate)
        diag = {
            "available_mean": available.float().mean(),
            "available_count": available.float().sum(),
            "feature_rms": [x.detach().float().square().mean().sqrt() for x in dct_features],
            "write_rms": [x.detach().float().square().mean().sqrt() for x in writes],
            "gate_mean": [x.detach().float().mean() for x in gates],
            "adapter_weight_rms": [a.proj.weight.detach().float().square().mean().sqrt() for a in self.dct_inject_adapters],
            "disabled": torch.tensor(float(bool(getattr(self, "disable_dct_evidence", False))), device=detail[0].device),
        }
        if self.dct_evidence_head is not None:
            diag["evidence_logits_low"] = self.dct_evidence_head(dct_features[0])
        return fused, diag, dct_features

    @staticmethod
    def _neutral_hcv(detail: list[Tensor]) -> dict[str, list[Tensor]]:
        """Fixed, non-learned HCV placeholders used only by the DHER ablation path.

        A probability of 0.5 carries no foreground/background preference and a
        zero signed effect carries no counterfactual update.  These tensors are
        derived only for shape/device compatibility and cannot learn HCV behavior.
        """
        proposals = [x.new_full((x.shape[0], 1, x.shape[2], x.shape[3]), 0.5) for x in detail[:3]]
        deltas = [x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3])) for x in detail[:3]]
        energies = [x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3])) for x in detail[:3]]
        return {"proposals": proposals, "deltas": deltas, "energies": energies}

    def _forward_e15_hr(
        self,
        image: Tensor,
        content: list[Tensor],
        detail_for_model: list[Tensor],
        qtr: dict | None,
        hcv: dict | None,
        anchor_content: list[Tensor],
        anchor_detail_for_model: list[Tensor],
        anchor_qtr: dict | None,
        anchor_hcv: dict | None,
    ) -> dict:
        if self.e15_decoder is None:
            raise RuntimeError("E15-HR forward requested without e15_decoder")
        old_decoded = self.decoder(anchor_content, anchor_detail_for_model)
        old_decoded, _ = self._apply_dher(old_decoded[0], old_decoded, anchor_detail_for_model, anchor_hcv, anchor_qtr)
        old_logits_low = self.mask_head(old_decoded[0])
        old_logits = F.interpolate(old_logits_low, size=image.shape[-2:], mode="bilinear", align_corners=False)

        e15_decoded = self.e15_decoder.coarse(content, detail_for_model, qtr, hcv)
        e15_d1_pre = e15_decoded[0]
        e15_decoded, dher_diag = self._apply_dher(e15_d1_pre, e15_decoded, detail_for_model, hcv, qtr)
        residual_full, boundary_logits, highres_diag = self.e15_decoder.reconstruct(image, e15_decoded[0], hcv)
        # E16 predicts a complete mask; the frozen old path is a teacher only.
        logits = residual_full if "e16_direct" in self.architecture else old_logits + residual_full
        log_var = F.interpolate(self.log_var_head(old_decoded[0]), size=image.shape[-2:], mode="bilinear", align_corners=False)
        old_aux_low = [head(f) for head, f in zip(self.aux_heads, old_decoded[1:])]
        old_aux = [F.interpolate(a, size=image.shape[-2:], mode="bilinear", align_corners=False) for a in old_aux_low]
        out = self._compat(logits, log_var, self.image_head(old_decoded[0]), old_aux, content)
        out["aux_logits_low"] = old_aux_low
        out["e15_old_logits"] = old_logits
        out["e15_old_logits_low"] = old_logits_low
        out["e15_residual_full"] = residual_full
        if "e16_direct" in self.architecture:
            out["direct_mask_logits"] = residual_full
            out["teacher_mask_logits"] = old_logits.detach()
        out["e15_boundary_logits"] = boundary_logits
        out["e15_d1_logits"] = self.e15_decoder.aux_d1(e15_decoded[0])
        out["e15_d2_logits"] = self.e15_decoder.aux_d2(e15_decoded[1])
        out["e15_d3_logits"] = self.e15_decoder.aux_d3(e15_decoded[2])
        out["e15_d1_pre_logits"] = self.mask_head(e15_d1_pre)
        out["e15_d1_pre_feature"] = e15_d1_pre
        out["e15_d1_refined_feature"] = e15_decoded[0]
        out["e15_decoded_shapes"] = [torch.tensor(x.shape[-2:], device=logits.device) for x in e15_decoded]
        out["e15_decoded_features"] = e15_decoded
        out["e15_detail_features"] = detail_for_model
        out["e15_highres"] = highres_diag
        if qtr is not None: out["qtr"] = qtr
        if hcv is not None: out["hcv2"] = hcv
        if anchor_qtr is not None: out["e15_anchor_qtr"] = anchor_qtr
        if anchor_hcv is not None: out["e15_anchor_hcv2"] = anchor_hcv
        if dher_diag is not None: out["dher"] = dher_diag
        return out

    def forward(self, image: Tensor, intervention_mask: Tensor | None = None, dct: Tensor | None = None, qtable: Tensor | None = None, dct_available: Tensor | None = None) -> dict:
        _ = intervention_mask
        content = self.content_encoder(image)
        if isinstance(self.forensic_stream, UnifiedForensicStream):
            raw_detail, detail, dct_diag, dct_features = (
                self.forensic_stream(
                    image, dct, qtable, dct_available,
                    disable_dct_evidence=bool(
                        self.disable_dct_evidence
                    ),
                )
            )
        elif getattr(self.forensic_stream, "requires_native_jpeg", False):
            if dct is None or qtable is None:
                raise ValueError("native JPEG forensic stream requires DCT and qtable tensors")
            detail = self.forensic_stream(image, dct, qtable, dct_available)
        elif isinstance(self.forensic_stream, DCTCueEncoder):
            if dct is None or qtable is None:
                raise ValueError("DCT forensic stream requires batch dct and qtable tensors")
            detail = self.forensic_stream(dct, qtable)
        else:
            detail = self.forensic_stream(image)
        if not isinstance(self.forensic_stream, UnifiedForensicStream):
            raw_detail = detail
            detail, dct_diag, dct_features = (
                self._inject_dct_evidence(
                    detail, dct, qtable, dct_available
                )
            )
        if self.e15_decoder is not None:
            if self.e15_anchor_content_encoder is None:
                raise RuntimeError("E15-HR requires e15_anchor_content_encoder")
            anchor_content = self.e15_anchor_content_encoder(image)
            anchor_qtr = self.qtr(image, anchor_content[:3], detail[:3]) if self.qtr is not None else None
            anchor_detail_for_model = [*anchor_qtr["qualified_detail"], *detail[3:]] if anchor_qtr is not None and "qualified_detail" in anchor_qtr else detail
            anchor_hcv = self.hcv2(anchor_content[:3], anchor_detail_for_model[:3], anchor_qtr) if self.hcv2 is not None else None
            qtr = self.qtr(image, content[:3], detail[:3]) if self.qtr is not None else None
            detail_for_model = [*qtr["qualified_detail"], *detail[3:]] if qtr is not None and "qualified_detail" in qtr else detail
            hcv = self.hcv2(content[:3], detail_for_model[:3], qtr) if self.hcv2 is not None else None
            out = self._forward_e15_hr(image, content, detail_for_model, qtr, hcv, anchor_content, anchor_detail_for_model, anchor_qtr, anchor_hcv)
            if dct_diag is not None:
                if "evidence_logits_low" in dct_diag:
                    out["dct_evidence_logits"] = F.interpolate(dct_diag["evidence_logits_low"], size=image.shape[-2:], mode="bilinear", align_corners=False)
                out["dct_evidence"] = dct_diag
            return out
        content_for_qtr = self.content_context(content) if self.content_context is not None else content
        qtr = self.qtr(image, content_for_qtr[:3], detail[:3]) if self.qtr is not None and not self.ablate_qtr else None
        detail_for_model = [*qtr["qualified_detail"], *detail[3:]] if qtr is not None and "qualified_detail" in qtr else detail
        hcv = self.hcv2(content[:3], detail_for_model[:3], qtr) if self.hcv2 is not None and not self.ablate_hcv else None
        if self.enable_balanced_residual_chain and hcv is not None:
            detail_for_model=[*hcv["verified_features"],detail_for_model[3]]
        qtr_aux_logits = [head(feat) for head, feat in zip(self.qtr_aux_heads, qtr["features"])] if qtr is not None and (self.enable_strict_serial_chain or self.enable_balanced_residual_chain) else []
        if self.enable_strict_serial_chain:
            if hcv is None or "verified_features" not in hcv:
                raise RuntimeError("strict serial chain requires HCV verified_features")
            verified = hcv["verified_features"]
            coarse = self.serial_coarse_adapter(verified[2])
            coarse = F.interpolate(coarse, size=detail[3].shape[-2:], mode="bilinear", align_corners=False)
            detail_for_model = [*verified, coarse]
        corr_diag = None
        if self.correspondence is not None:
            d4 = self.decoder.stage4(content[3], detail_for_model[3])
            d3 = self.decoder.stage3(content[2], detail_for_model[2], d4)
            d2_base = self.decoder.stage2(content[1], detail_for_model[1], d3)
            corr_write, corr_diag = self.correspondence(content[1])
            if corr_write.shape[-2:] != d2_base.shape[-2:]:
                corr_write = F.interpolate(corr_write, size=d2_base.shape[-2:], mode="bilinear", align_corners=False)
                corr_diag["write"] = corr_write
            d2 = d2_base + corr_write
            d1 = self.decoder.stage1(content[0], detail_for_model[0], d2)
            decoded = [d1, d2, d3, d4]
        else:
            decoded = self.decoder(content, detail_for_model)
        dher_diag = None
        dher_hcv = self._neutral_hcv(detail_for_model) if self.ablate_hcv and not self.ablate_dher else hcv
        if self.dher is not None and dher_hcv is not None and not self.ablate_dher:
            if isinstance(self.dher, DHERv21):
                base_logits_d1 = self.mask_head(decoded[0]).detach()
                d1, dher_diag = self.dher(decoded[0], detail_for_model[:2], dher_hcv, qtr, base_logits_d1)
            elif isinstance(self.dher, DHERv2):
                d1, dher_diag = self.dher(decoded[0], detail_for_model[:2], dher_hcv, qtr)
            else:
                d1, dher_diag = self.dher(decoded[0], detail_for_model[0], dher_hcv, qtr)
            decoded = [d1, *decoded[1:]]
        if self.p2_preserve is not None:
            decoded = [self.p2_preserve(decoded[0], detail_for_model[0]), *decoded[1:]]
        splicing_region_diag = None
        if self.splicing_region_affinity is not None:
            if dct_features is None:
                raise RuntimeError("splicing region affinity requires the parallel DCT branch")
            available = dct_available
            if available is None:
                available = torch.ones((image.shape[0], 1), dtype=image.dtype, device=image.device)
            refined_d1, splicing_region_diag = self.splicing_region_affinity(
                decoded[0], raw_detail[0], dct_features[0], available,
                disabled=bool(getattr(self, "disable_splicing_region", False)),
            )
            decoded = [refined_d1, *decoded[1:]]
        dher_calibrated_diag = None
        if self.dher_calibrated_writeback is not None and hcv is not None and dher_diag is not None:
            refined_d1, dher_calibrated_diag = self.dher_calibrated_writeback.apply_feature(
                decoded[0], hcv, dher_diag, disabled=bool(getattr(self, "disable_dher_calibrated_writeback", False))
            )
            decoded = [refined_d1, *decoded[1:]]
        logits_low = self.mask_head(decoded[0])
        dher_visibility_diag = None
        if self.dher_visibility is not None and hcv is not None and dher_diag is not None and not bool(getattr(self, "disable_dher_visibility", False)):
            logits_low, dher_visibility_diag = self.dher_visibility(logits_low, hcv, dher_diag)
        if self.dher_calibrated_writeback is not None and hcv is not None and dher_diag is not None:
            logits_low, logit_diag = self.dher_calibrated_writeback.apply_logit(
                logits_low, hcv, dher_diag, disabled=bool(getattr(self, "disable_dher_calibrated_writeback", False))
            )
            if dher_calibrated_diag is None:
                dher_calibrated_diag = {}
            dher_calibrated_diag.update(logit_diag)
        region_residual = None
        if self.region_refine is not None:
            logits_low, region_residual = self.region_refine(decoded[0], logits_low, hcv)
        refine_delta = None
        if self.mask_refine is not None:
            logits_low, refine_delta = self.mask_refine(decoded[0], logits_low)
        logits = F.interpolate(logits_low, size=image.shape[-2:], mode="bilinear", align_corners=False)
        log_var = F.interpolate(self.log_var_head(decoded[0]), size=image.shape[-2:], mode="bilinear", align_corners=False)
        aux_low = [head(f) for head, f in zip(self.aux_heads, decoded[1:])]
        aux = [F.interpolate(a, size=image.shape[-2:], mode="bilinear", align_corners=False) for a in aux_low]
        out = self._compat(logits, log_var, self.image_head(decoded[0]), aux, content)
        out["aux_logits_low"] = aux_low
        if qtr is not None: out["qtr"] = qtr
        if qtr_aux_logits: out["qtr_aux_logits"] = qtr_aux_logits
        if hcv is not None: out["hcv2"] = hcv
        if dct_diag is not None:
            if "evidence_logits_low" in dct_diag:
                out["dct_evidence_logits"] = F.interpolate(dct_diag["evidence_logits_low"], size=image.shape[-2:], mode="bilinear", align_corners=False)
            out["dct_evidence"] = dct_diag
        if dher_diag is not None: out["dher"] = dher_diag
        if dher_visibility_diag is not None: out["dher_visibility"] = dher_visibility_diag
        if dher_calibrated_diag is not None: out["dher_calibrated_writeback"] = dher_calibrated_diag
        if splicing_region_diag is not None: out["splicing_region"] = splicing_region_diag
        if corr_diag is not None: out["correspondence"] = corr_diag
        if self.p2_preserve is not None: out["p2_preserve_enabled"] = torch.ones((), device=logits.device)
        if self.content_context is not None: out["content_context_enabled"] = torch.ones((), device=logits.device)
        if self.mask_refine is not None:
            out["mask_refine_delta"] = refine_delta
            out["mask_refine_alpha"] = self.mask_refine.alpha
        if self.region_refine is not None:
            out["region_refine_residual"] = region_residual
        if getattr(self.decoder, "forensic_reconstruction", None) is not None:
            out["decoder_forensic_reconstruction_enabled"] = torch.ones((), device=logits.device)
        if getattr(self.forensic_stream, "fft_prior_fusion", None) is not None:
            out["fft_prior_enabled"] = torch.ones((), device=logits.device)
        if getattr(self.forensic_stream, "srm_prior_fusion", None) is not None:
            out["srm_prior_enabled"] = torch.ones((), device=logits.device)
        return out


def build_final_model(config: dict) -> FinalReTraceNet:
    model = FinalReTraceNet(
        architecture=str(config.get("architecture", "retracenet_b0")),
        decoder_channels=int(config.get("decoder_channels", 128)),
        residual_kernels=int(config.get("residual_kernels", 8)),
        forensic_channels=tuple(config.get("forensic_channels", (64,128,256,512))),
        pretrained_path=config.get("pretrained_path"),
        freeze_encoder_bn=bool(config.get("freeze_encoder_bn", True)),
        content_backbone=str(config.get("content_backbone", config.get("encoder_backbone", "resnet50"))),
        content_channels=tuple(config.get("content_channels", (96,192,384,768) if str(config.get("content_backbone", config.get("encoder_backbone", "resnet50"))) == "convnext_tiny" else (256,512,1024,2048))),
        convnext_pretrained=bool(config.get("convnext_pretrained", True)),
        forensic_stream_type=str(config.get("forensic_stream_type", config.get("forensic_encoder", "residual"))),
        dct_pretrained_path=config.get("dct_pretrained_path"),
        freeze_forensic_bn=bool(config.get("freeze_forensic_bn", False)),
        p2_preserve_channels=int(config.get("p2_preserve_channels", 64)),
        enable_mrfe=bool(config.get("enable_mrfe", False)),
        enable_fft_prior=bool(config.get("enable_fft_prior", False)),
        fft_prior_cutoff=float(config.get("fft_prior_cutoff", 0.18)),
        enable_srm_prior=bool(config.get("enable_srm_prior", False)),
        enable_decoder_forensic_reconstruction=bool(config.get("enable_decoder_forensic_reconstruction", False)),
        decoder_forensic_reconstruction_proj_channels=int(config.get("decoder_forensic_reconstruction_proj_channels", 32)),
        hrnet_model_name=str(config.get("hrnet_model_name", "hrnet_w18")),
        hrnet_pretrained=bool(config.get("hrnet_pretrained", True)),
        enable_dct_parallel=bool(config.get("enable_dct_parallel", False)),
        dct_injection_zero_init=bool(config.get("dct_injection_zero_init", True)),
        enable_dct_gate=bool(config.get("enable_dct_gate", False)),
        dct_gate_bias=float(config.get("dct_gate_bias", -4.0)),
        enable_dct_evidence_head=bool(config.get("enable_dct_evidence_head", False)),
        enable_splicing_region_affinity=bool(config.get("enable_splicing_region_affinity", False)),
        splicing_affinity_channels=int(config.get("splicing_affinity_channels", 32)),
        enable_dher_visibility=bool(config.get("enable_dher_visibility", False)),
        enable_dher_calibrated_writeback=bool(config.get("enable_dher_calibrated_writeback", False)),
        enable_effective_mechanism_chain=bool(config.get("enable_effective_mechanism_chain", False)),
        enable_strict_serial_chain=bool(config.get("enable_strict_serial_chain", False)),
        enable_balanced_residual_chain=bool(config.get("enable_balanced_residual_chain", False)),
    )
    if config.get("enable_region_consistency", False):
        if config.get("enable_splicing_region_affinity", False):
            raise ValueError("Do not combine HCV consistency and E23 decoder affinity in this experiment")
        from .hcv_region_consistency import attach_region_consistency
        attach_region_consistency(model.hcv2)
    return model
