import torch

from retrace.blocks import ConstrainedResidualBank
from retrace.losses import ReTraceLoss
from retrace.model import ReTraceNet


def tiny_model() -> ReTraceNet:
    return ReTraceNet(
        base_channels=8,
        decoder_channels=16,
        depths=(1, 1, 1, 1),
        recycle_steps=1,
        residual_kernels=4,
        drop_path=0.0,
    )


def test_constrained_kernels_sum_to_zero() -> None:
    bank = ConstrainedResidualBank(4)
    sums = bank.constrained_weight().sum(dim=(1, 2, 3))
    assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-6)


def test_model_shapes_and_finite_loss() -> None:
    model = tiny_model()
    image = torch.rand(2, 3, 64, 80)
    mask = torch.zeros(2, 1, 64, 80)
    mask[0, :, 16:32, 20:45] = 1
    output = model(image)
    assert output["mask_logits"].shape == mask.shape
    assert output["log_var"].shape == mask.shape
    assert output["image_logit"].shape == (2, 1)
    assert output["proposal_logits"].shape == mask.shape
    assert output["counterfactual_logits"].shape == mask.shape
    assert output["causal_effect"].shape == mask.shape
    assert len(output["ledger_logits"]) == 4
    assert output["scale_reliability"].shape == (2, 4)
    criterion = ReTraceLoss(
        weights={
            "localization": 1.0,
            "dice": 0.5,
            "boundary": 0.2,
            "deep_supervision": 0.3,
            "proposal": 0.1,
            "counterfactual": 0.2,
            "ledger": 0.15,
            "intervention_locality": 0.03,
            "orthogonality": 0.05,
            "image": 0.2,
            "degradation_consistency": 0.1,
            "variance_regularization": 0.01,
        },
        boundary_kernel=3,
    )
    losses = criterion(output, mask)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.residual_bank.raw.grad is not None
