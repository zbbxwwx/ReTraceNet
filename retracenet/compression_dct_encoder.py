"""DCT/qtable cue encoder; see THIRD_PARTY_NOTICES.md."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import LayerNorm2d

BN_MOMENTUM = 0.01


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches: int, num_blocks: list[int], num_inchannels: list[int], num_channels: list[int], fuse_method: str, multi_scale_output: bool = True) -> None:
        super().__init__()
        self.num_inchannels = list(num_inchannels)
        self.num_branches = int(num_branches)
        self.fuse_method = fuse_method
        self.multi_scale_output = bool(multi_scale_output)
        self.branches = self._make_branches(num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index: int, num_blocks: list[int], num_channels: list[int], stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index]:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index], momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [BasicBlock(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index]
        for _ in range(1, num_blocks[branch_index]):
            layers.append(BasicBlock(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_blocks: list[int], num_channels: list[int]) -> nn.ModuleList:
        return nn.ModuleList(self._make_one_branch(i, num_blocks, num_channels) for i in range(self.num_branches))

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        if self.num_branches == 1:
            return None
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        outchannels = self.num_inchannels[i] if k == i - j - 1 else self.num_inchannels[j]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(self.num_inchannels[j], outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True) if k != i - j - 1 else nn.Identity(),
                        ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> list[int]:
        return self.num_inchannels

    def forward(self, x: list[Tensor]) -> list[Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        fused = []
        assert self.fuse_layers is not None
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(fuse_layer[j](x[j]), size=x[i].shape[-2:], mode="bilinear", align_corners=False)
                else:
                    y = y + fuse_layer[j](x[j])
            fused.append(self.relu(y))
        return fused


class DCTCueEncoder(nn.Module):
    """DCT volume and quantization-table encoder for four scales."""

    channels = (64, 128, 256, 512)

    def __init__(self, channels: tuple[int, int, int, int] = (64, 128, 256, 512), pretrained_path: str | None = None, freeze_bn: bool = False) -> None:
        super().__init__()
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(21, 64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.dc_layer2 = self._make_layer(512, 96, blocks=4, stride=1)
        self.dc_transition2 = self._make_transition_layer([96], [96, 192])
        self.dc_stage3, pre_stage_channels = self._make_stage(num_modules=3, num_branches=2, num_blocks=[4, 4], num_channels=[96, 192], fuse_method="SUM", num_inchannels=[96, 192])
        self.dc_transition3 = self._make_transition_layer(pre_stage_channels, [96, 192, 384])
        self.dc_stage4, _ = self._make_stage(num_modules=2, num_branches=3, num_blocks=[4, 4, 4], num_channels=[96, 192, 384], fuse_method="SUM", num_inchannels=[96, 192, 384])
        c1, c2, c3, c4 = channels
        self.output_projections = nn.ModuleList([
            self._projection(96, c1),
            self._projection(96, c2),
            self._projection(192, c3),
            self._projection(384, c4),
        ])
        self.freeze_bn = bool(freeze_bn)
        if pretrained_path:
            self.load_dct_pretrained(pretrained_path)
        if self.freeze_bn:
            self._freeze_bn()

    @staticmethod
    def _projection(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
        layers: list[nn.Module] = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer: list[int], num_channels_cur_layer: list[int]) -> nn.ModuleList:
        transition_layers = []
        num_branches_pre = len(num_channels_pre_layer)
        for i, cur_channels in enumerate(num_channels_cur_layer):
            if i < num_branches_pre:
                if cur_channels != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], cur_channels, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur_channels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = cur_channels if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules: int, num_branches: int, num_blocks: list[int], num_channels: list[int], fuse_method: str, num_inchannels: list[int], multi_scale_output: bool = True) -> tuple[nn.Sequential, list[int]]:
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i != num_modules - 1
            modules.append(HighResolutionModule(num_branches, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

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

    def load_dct_pretrained(self, pretrained_path: str) -> dict[str, int]:
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(f"DCT pretrained checkpoint not found: {pretrained_path}")
        checkpoint = torch.load(path, map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        source = {str(k).replace("module.", ""): v for k, v in source.items()}
        target = self.state_dict()
        filtered = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape) and k.startswith("dc_")}
        result = self.load_state_dict(filtered, strict=False)
        dc_keys = [k for k in target if k.startswith("dc_")]
        missing_dc = [k for k in dc_keys if k not in filtered and not k.endswith("num_batches_tracked")]
        print(f"DCTCueEncoder loaded DCT pretrained keys: {len(filtered)} from {pretrained_path}; missing_dc={len(missing_dc)} adapters_missing={sum(k.startswith('output_projections.') for k in result.missing_keys)}")
        if missing_dc:
            print(f"DCTCueEncoder missing DCT examples: {missing_dc[:8]}")
        return {"loaded": len(filtered), "missing_dc": len(missing_dc)}

    def forward(self, dct: Tensor, qtable: Tensor) -> list[Tensor]:
        if dct.shape[1] != 21:
            raise ValueError(f"expected DCTvol with 21 channels, got {tuple(dct.shape)}")
        if qtable.ndim == 3:
            qtable = qtable.unsqueeze(1)
        if qtable.ndim != 4 or qtable.shape[-2:] != (8, 8):
            raise ValueError(f"expected qtable shape [B,1,8,8] or [B,8,8], got {tuple(qtable.shape)}")
        x = self.dc_layer0_dil(dct.float())
        x = self.dc_layer1_tail(x)
        b, c, h, w = x.shape
        h8, w8 = h // 8, w // 8
        if h8 == 0 or w8 == 0:
            raise ValueError(f"DCT feature map too small for 8x8 block folding: {tuple(x.shape)}")
        x = x[..., : h8 * 8, : w8 * 8]
        x_blocks = x.reshape(b, c, h8, 8, w8, 8).permute(0, 1, 3, 5, 2, 4)
        q = qtable[:, :1].to(dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)
        x0 = x_blocks.reshape(b, 64 * c, h8, w8)
        x1 = (x_blocks * q).reshape(b, 64 * c, h8, w8)
        x = self.dc_layer2(torch.cat([x0, x1], dim=1))
        x_list = [self.dc_transition2[i](x) if self.dc_transition2[i] is not None else x for i in range(2)]
        y_list = self.dc_stage3(x_list)
        x_list = [self.dc_transition3[i](y_list[-1]) if self.dc_transition3[i] is not None else y_list[i] for i in range(3)]
        dc_list = self.dc_stage4(x_list)
        n1 = self.output_projections[0](F.interpolate(dc_list[0], scale_factor=2.0, mode="bilinear", align_corners=False))
        n2 = self.output_projections[1](dc_list[0])
        n3 = self.output_projections[2](dc_list[1])
        n4 = self.output_projections[3](dc_list[2])
        return [n1, n2, n3, n4]
