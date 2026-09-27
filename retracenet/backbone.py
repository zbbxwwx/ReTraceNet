"""Content feature encoder for the v2 B0 base."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResNet50ContentEncoder(nn.Module):
    """ResNet-50 C2-C5 feature pyramid."""

    channels = (256, 512, 1024, 2048)

    def __init__(
        self,
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected:
                print(f"ResNet50ContentEncoder ignored unexpected keys: {len(unexpected)}")
            if missing:
                print(f"ResNet50ContentEncoder missing keys: {len(missing)}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.imagenet_normalize = bool(imagenet_normalize)
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

    def forward(self, image: Tensor) -> list[Tensor]:
        x = image
        if self.imagenet_normalize:
            x = (x - self.mean) / self.std
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]

class ResNet101ContentEncoder(ResNet50ContentEncoder):
    """ResNet-101 C2-C5 feature pyramid with the same channels as ResNet-50."""

    channels = (256, 512, 1024, 2048)

    def __init__(
        self,
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        from torchvision.models import resnet101

        backbone = resnet101(weights=None)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected:
                print(f"ResNet101ContentEncoder ignored unexpected keys: {len(unexpected)}")
            if missing:
                print(f"ResNet101ContentEncoder missing keys: {len(missing)}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.imagenet_normalize = bool(imagenet_normalize)
        self.freeze_bn = bool(freeze_bn)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if self.freeze_bn:
            self._freeze_bn()


class ResNeXt101_32x8dContentEncoder(ResNet50ContentEncoder):
    """ResNeXt-101 32x8d C2-C5 feature pyramid with ResNet-compatible channels."""

    channels = (256, 512, 1024, 2048)

    def __init__(
        self,
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        freeze_bn: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        from torchvision.models import resnext101_32x8d

        backbone = resnext101_32x8d(weights=None)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected:
                print(f"ResNeXt101_32x8dContentEncoder ignored unexpected keys: {len(unexpected)}")
            if missing:
                print(f"ResNeXt101_32x8dContentEncoder missing keys: {len(missing)}")
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.imagenet_normalize = bool(imagenet_normalize)
        self.freeze_bn = bool(freeze_bn)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if self.freeze_bn:
            self._freeze_bn()


class ConvNeXtTinyContentEncoder(nn.Module):
    """ConvNeXt-Tiny feature pyramid: strides 4, 8, 16, 32."""

    channels = (96, 192, 384, 768)

    def __init__(
        self,
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = None
        if pretrained and not pretrained_path:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, Mapping) and "model" in state:
                state = state["model"]
            state = {str(k).replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected:
                print(f"ConvNeXtTinyContentEncoder ignored unexpected keys: {len(unexpected)}")
            if missing:
                print(f"ConvNeXtTinyContentEncoder missing keys: {len(missing)}")
        self.features = backbone.features
        self.imagenet_normalize = bool(imagenet_normalize)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: Tensor) -> list[Tensor]:
        x = image
        if self.imagenet_normalize:
            x = (x - self.mean) / self.std
        x = self.features[0](x)
        c1 = self.features[1](x)
        x = self.features[2](c1)
        c2 = self.features[3](x)
        x = self.features[4](c2)
        c3 = self.features[5](x)
        x = self.features[6](c3)
        c4 = self.features[7](x)
        return [c1, c2, c3, c4]


class HRNetW18ContentEncoder(nn.Module):
    """HRNet-W18 ImageNet feature pyramid adapted to C1-C4 strides 4, 8, 16, 32."""

    channels = (256, 512, 1024, 2048)

    def __init__(
        self,
        model_name: str = "hrnet_w18",
        pretrained_path: str | None = None,
        imagenet_normalize: bool = True,
        pretrained: bool = True,
        out_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=bool(pretrained and not pretrained_path),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.pretrained_load_report = {
            "path": str(pretrained_path or ""),
            "checkpoint_keys": 0,
            "matched_keys": 0,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, Mapping) and "model" in state:
                state = state["model"]
            state = {str(k).replace("module.", "").replace("model.", ""): v for k, v in state.items()}
            target_state = self.backbone.state_dict()
            matched = [
                k for k, v in state.items()
                if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
            ]
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            self.pretrained_load_report = {
                "path": str(pretrained_path),
                "checkpoint_keys": len(state),
                "matched_keys": len(matched),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
            if unexpected:
                print(f"HRNetW18ContentEncoder ignored unexpected keys: {len(unexpected)}")
            if missing:
                print(f"HRNetW18ContentEncoder missing keys: {len(missing)}")
            print(f"HRNetW18ContentEncoder loaded offline checkpoint keys: {len(matched)}")
        raw_channels = [int(d["num_chs"]) for d in self.backbone.feature_info.get_dicts()]
        selected_channels = [raw_channels[1], raw_channels[2], raw_channels[3], raw_channels[3]]
        self.proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(inp, out, 1, bias=False),
                nn.GroupNorm(1, out),
                nn.GELU(),
            )
            for inp, out in zip(selected_channels, out_channels)
        )
        self.imagenet_normalize = bool(imagenet_normalize)
        self.out_channels = tuple(int(c) for c in out_channels)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: Tensor) -> list[Tensor]:
        x = image
        if self.imagenet_normalize:
            x = (x - self.mean) / self.std
        feats = self.backbone(x)
        h, w = image.shape[-2:]
        targets = [
            (max(1, h // 4), max(1, w // 4)),
            (max(1, h // 8), max(1, w // 8)),
            (max(1, h // 16), max(1, w // 16)),
            (max(1, h // 32), max(1, w // 32)),
        ]
        raw = [feats[1], feats[2], feats[3], F.avg_pool2d(feats[3], kernel_size=2, stride=2)]
        out = []
        for feat, target_size, proj in zip(raw, targets, self.proj):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            out.append(proj(feat))
        return out
