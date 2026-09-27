from pathlib import Path
import ast
import shutil
import time

root = Path.cwd()
main = root / "retracenet/final_modules.py"
old_dct = root / "retracenet/cat_dct_stream.py"
new_dct = root / "retracenet/compression_dct_encoder.py"
mit = root / "tools/run_imd_mit_full_v3.py"

for p in (main, old_dct, mit):
    if not p.is_file():
        raise SystemExit(f"Missing: {p}")

source = main.read_text()

if "class UnifiedForensicStream" in source:
    raise SystemExit("Already integrated; stop")

def replace_once(s, old, new):
    count = s.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one match, got {count}: {old[:80]}"
        )
    return s.replace(old, new, 1)

# Preserve the original implementation and weights.
backup = root / (
    "backup_cue_integration_"
    + time.strftime("%Y%m%d_%H%M%S")
)
backup.mkdir()
shutil.copy2(main, backup / "final_modules.py")
shutil.copy2(mit, backup / "run_imd_mit_full_v3.py")

# Create the functionally named DCT encoder.
encoder = old_dct.read_text()
encoder = encoder.replace(
    "DCTCompressionForensicStream",
    "DCTCueEncoder"
)
encoder = encoder.replace(
    "CAT-style DCT/qtable forensic stream for ReTraceNet.",
    "DCT/qtable cue encoder; see THIRD_PARTY_NOTICES.md."
)
encoder = encoder.replace(
    "CAT DCTvol+qtable branch adapted to ReTraceNet N1-N4 channels.",
    "DCT volume and quantization-table encoder for four scales."
)

wrapper = '''
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

'''

source = replace_once(
    source,
    "from .cat_dct_stream import DCTCompressionForensicStream",
    "from .compression_dct_encoder import DCTCueEncoder"
)
source = source.replace(
    "DCTCompressionForensicStream", "DCTCueEncoder"
)
source = source.replace('"cat_dct"', '"dct_cue"')

source = replace_once(
    source,
    "class QualityTraceQualification(nn.Module):",
    wrapper + "class QualityTraceQualification(nn.Module):"
)

old = (
    '        if self.forensic_stream_type in '
    '{"residual", "noise", "lightweight"}:'
)
new = '''
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
'''
source = replace_once(source, old, new)

source = replace_once(
    source,
    '        if getattr(self.forensic_stream, "requires_native_jpeg", False):',
    '''        if isinstance(self.forensic_stream, UnifiedForensicStream):
            raw_detail, detail, dct_diag, dct_features = (
                self.forensic_stream(
                    image, dct, qtable, dct_available,
                    disable_dct_evidence=bool(
                        self.disable_dct_evidence
                    ),
                )
            )
        elif getattr(self.forensic_stream, "requires_native_jpeg", False):'''
)

source = replace_once(
    source,
    '''        raw_detail = detail
        detail, dct_diag, dct_features = self._inject_dct_evidence(detail, dct, qtable, dct_available)''',
    '''        if not isinstance(self.forensic_stream, UnifiedForensicStream):
            raw_detail = detail
            detail, dct_diag, dct_features = (
                self._inject_dct_evidence(
                    detail, dct, qtable, dct_available
                )
            )'''
)

# Avoid reloading pretrained MiT weights when converting
# a checkpoint that already contains all model weights.
mit_source = mit.read_text()
mit_source = replace_once(
    mit_source,
    '''    def __init__(self):
        super().__init__()
        from lib.models.cmx.encoders.dual_segformer import mit_b2''',
    '''    def __init__(self, initialize_pretrained=True):
        super().__init__()
        from lib.models.cmx.encoders.dual_segformer import mit_b2'''
)
mit_source = replace_once(
    mit_source,
    '        self.net.init_weights_onepath(str(MIT_WEIGHTS))',
    '''        if initialize_pretrained:
            self.net.init_weights_onepath(str(MIT_WEIGHTS))'''
)
mit_source = replace_once(
    mit_source,
    '    model.content_encoder = MiTB2ContentCompat()',
    '''    model.content_encoder = MiTB2ContentCompat(
        initialize_pretrained=bool(
            cfg.get("initialize_mit_pretrained", True)
        )
    )'''
)
mit_source = replace_once(
    mit_source,
    "    for branch in (getattr(model,'content_encoder',None), getattr(model,'dct_branch',None)):",
    """    for branch in (
        getattr(model, 'content_encoder', None),
        getattr(model, 'dct_branch', None),
        getattr(
            getattr(model, 'forensic_stream', None),
            'dct_encoder', None
        ),
    ):"""
)

# Validate syntax before overwriting source files.
for name, text in [
    (str(main), source),
    (str(new_dct), encoder),
    (str(mit), mit_source),
]:
    ast.parse(text, filename=name)

new_dct.write_text(encoder)
main.write_text(source)
mit.write_text(mit_source)

notice = root / "THIRD_PARTY_NOTICES.md"
if not notice.exists():
    notice.write_text(
        "# Third-party components\n\n"
        "The DCT Cue encoder in "
        "retracenet/compression_dct_encoder.py "
        "is adapted from the CAT-Net DCT branch. "
        "Its double-JPEG pretrained weights originate "
        "from CAT-Net. Include the original citation "
        "and applicable license when distributing.\n"
    )

print("PATCH PASS")
print("Backup:", backup)
print("Unified Forensic Stream created")
