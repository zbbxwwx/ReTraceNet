"""Optional post-verification refinement, owned by HCV; no encoder changes."""
import torch
from torch import nn
import torch.nn.functional as F


class SignedRegionConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
        self.refinement = nn.Sequential(
            nn.Conv2d(6, 8, 1), nn.GELU(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.GELU(),
            nn.Conv2d(8, 1, 1), nn.Tanh(),
        )

    def forward(self, delta, proposal):
        near = F.avg_pool2d(delta, 3, 1, 1, count_include_pad=False)
        context = F.avg_pool2d(delta, 7, 1, 3, count_include_pad=False)
        disagreement = F.avg_pool2d((delta - context).abs(), 7, 1, 3, count_include_pad=False)
        cues = torch.cat([delta, near, context, delta-context, disagreement, proposal], 1)
        return delta + self.scale * self.refinement(cues)


def attach_region_consistency(hcv):
    """Preserve HCV type, signature, keys and shapes; attach one output hook."""
    if hcv is None:
        raise ValueError('region consistency requires HCV')
    if hasattr(hcv, 'region_consistency'):
        raise ValueError('region consistency already attached')
    # E5/E21 DHER consumes delta1/delta2; delta3 must not gain dead parameters.
    hcv.region_consistency = nn.ModuleList([SignedRegionConsistency() for _ in range(2)])
    hcv.enable_region_consistency = True
    hcv.register_forward_hook(_refine_verified_effects)
    return hcv


def _refine_verified_effects(hcv, inputs, output):
    if not hcv.enable_region_consistency:
        return output
    result = dict(output)
    result['deltas'] = [*output['deltas']]
    for i, layer in enumerate(hcv.region_consistency):
        result['deltas'][i] = layer(output['deltas'][i], output['proposals'][i])
    return result
