# ReTraceNet

ReTraceNet is a research implementation for image manipulation localization. The current model uses two top-level streams:

- a MiT-B2 content encoder;
- a unified forensic stream that combines residual and compression/DCT cues through balanced multi-scale fusion.

The resulting feature pyramids are processed by QTR, HCV, a progressive decoder, and DHER to produce a pixel-level manipulation mask and an image-level score.

## Current implementation

The maintained implementation is under `retracenet/`. The verified IMD training entry point is `tools/run_imd_mit_full_v3.py`.

The unified forensic refactor preserves checkpoint compatibility with the previous implementation. Using the IMD2020 epoch-29 checkpoint, strict loading succeeds with 140,211,179 parameters. On the equivalence test input, the four-scale residual, DCT, and fused forensic features, final mask, and image score all have a maximum absolute difference of `0`.

## Repository layout

```text
retracenet/                         current ReTraceNet model components
tools/run_imd_mit_full_v3.py        verified IMD training entry point
tools/apply_cue_integration.py      unified forensic stream migration helper
configs/                            experiment configurations
tests/                              unit and smoke tests
docs/                               architecture and experiment notes
paper/                              manuscript sources
```

The earlier `retrace/` implementation and historical experiment tools are retained for reproducibility. New development should target `retracenet/`.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pretrained weights, datasets, checkpoints, and experiment outputs are intentionally excluded from this repository. Configure their paths locally before training.

Some historical experiment configurations preserve machine-specific absolute paths to document the original runs. Replace those paths with locations in your environment before reuse.

## Training and evaluation

The verified IMD entry point is:

```bash
python tools/run_imd_mit_full_v3.py --help
```

Evaluation scripts and dataset-specific runners are available under `tools/`. Dataset paths and pretrained initialization paths must be supplied for the target environment. Test-set oracle thresholds are diagnostic only; formal results should use thresholds selected on the source validation split.

## Documentation

- [Architecture notes](docs/architecture_zh-CN.md)
- [Validation record](docs/validation.md)
- [Current code audit](docs/code_audit.md)
- [Experiment history](docs/experiment_history.md)
- [Experiment plan](docs/experiment_plan.md)
- [Third-party components](THIRD_PARTY_NOTICES.md)

## Third-party components

The DCT cue encoder is adapted from the CAT-Net DCT branch and may use its double-JPEG pretrained weights. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), cite the original work, and comply with the upstream license when distributing those components or weights.

## License

ReTraceNet source code is released under the MIT License. Dataset licenses, pretrained weights, and third-party components remain governed by their respective terms.
