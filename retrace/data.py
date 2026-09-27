"""Unified manifest dataset for image manipulation localization."""

from __future__ import annotations

import csv
import io
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass(frozen=True)
class Record:
    image: Path
    mask: Path | None
    split: str
    dataset: str
    label: int


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_manifest(path: str | Path, root: str | Path, split: str) -> list[Record]:
    root = Path(root)
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split", "train") != split:
                continue
            mask_value = row.get("mask", "").strip()
            label = int(row.get("label", "1" if mask_value else "0"))
            records.append(
                Record(
                    image=_resolve(root, row["image"]),
                    mask=_resolve(root, mask_value) if mask_value else None,
                    split=split,
                    dataset=row.get("dataset", "unknown"),
                    label=label,
                )
            )
    if not records:
        raise ValueError(f"no records for split={split!r} in {path}")
    return records


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    array = np.ascontiguousarray(array.transpose(2, 0, 1)) / 255.0
    return torch.from_numpy(array)


class ManipulationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[Record],
        image_size: int = 512,
        training: bool = False,
        jpeg_probability: float = 0.0,
        jpeg_quality: tuple[int, int] | list[int] = (45, 96),
        blur_probability: float = 0.0,
        color_jitter: float = 0.0,
        positive_crop_probability: float = 0.0,
        boundary_crop_probability: float = 0.0,
        min_positive_crop_ratio: float = 0.0,
        crop_retries: int = 10,
        dct_forensic: bool = False,
        dct_jpeg_quality: int = 100,
        dct_qtable_from_source: bool = True,
        dct_cat_root: str | Path | None = None,
        dct_source_mode: str = "reencoded",
    ) -> None:
        self.records = records
        self.image_size = int(image_size)
        self.training = training
        self.jpeg_probability = float(jpeg_probability)
        self.jpeg_quality = tuple(int(x) for x in jpeg_quality)
        self.blur_probability = float(blur_probability)
        self.color_jitter = float(color_jitter)
        self.positive_crop_probability = float(positive_crop_probability)
        self.boundary_crop_probability = float(boundary_crop_probability)
        self.min_positive_crop_ratio = float(min_positive_crop_ratio)
        self.crop_retries = int(crop_retries)
        self.dct_forensic = bool(dct_forensic)
        self.dct_jpeg_quality = int(dct_jpeg_quality)
        self.dct_qtable_from_source = bool(dct_qtable_from_source)
        self.dct_cat_root = Path(dct_cat_root) if dct_cat_root else Path("/group4/zouwenxuan/iml_generalization/data/CATNET_CASIA")
        self.dct_source_mode = str(dct_source_mode)

    def __len__(self) -> int:
        return len(self.records)

    def _joint_geometry(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        target = self.image_size
        if self.training:
            scale = random.uniform(0.75, 1.5)
            shorter = max(1, min(image.size))
            ratio = target * scale / shorter
            new_size = (
                max(target, round(image.width * ratio)),
                max(target, round(image.height * ratio)),
            )
            image = image.resize(new_size, Image.Resampling.BILINEAR)
            mask = mask.resize(new_size, Image.Resampling.NEAREST)

            def crop_at(left: int, top: int) -> tuple[Image.Image, Image.Image]:
                left = max(0, min(left, image.width - target))
                top = max(0, min(top, image.height - target))
                box = (left, top, left + target, top + target)
                return image.crop(box), mask.crop(box)

            def crop_once() -> tuple[Image.Image, Image.Image]:
                return crop_at(
                    random.randint(0, image.width - target),
                    random.randint(0, image.height - target),
                )

            def foreground_ratio(candidate: Image.Image) -> float:
                return float((np.asarray(candidate, dtype=np.uint8) > 0).mean())

            def crop_centered_on(points: np.ndarray) -> tuple[Image.Image, Image.Image]:
                y, x = points[random.randrange(len(points))]
                jitter = target // 4
                left = int(x) - target // 2 + random.randint(-jitter, jitter)
                top = int(y) - target // 2 + random.randint(-jitter, jitter)
                return crop_at(left, top)

            mask_array = np.asarray(mask, dtype=np.uint8) > 0
            foreground = np.argwhere(mask_array)
            image_crop, mask_crop = crop_once()
            if len(foreground) > 0:
                boundary = foreground
                if self.boundary_crop_probability > 0:
                    padded = np.pad(mask_array, 1, mode="constant", constant_values=False)
                    center = padded[1:-1, 1:-1]
                    eroded = (
                        padded[:-2, 1:-1] & padded[2:, 1:-1]
                        & padded[1:-1, :-2] & padded[1:-1, 2:]
                        & center
                    )
                    boundary_points = np.argwhere(center & ~eroded)
                    if len(boundary_points) > 0:
                        boundary = boundary_points
                want_positive = random.random() < self.positive_crop_probability
                want_boundary = random.random() < self.boundary_crop_probability
                best_image, best_mask = image_crop, mask_crop
                best_ratio = foreground_ratio(mask_crop)
                tries = max(1, self.crop_retries)
                for attempt in range(tries):
                    if want_boundary:
                        candidate_image, candidate_mask = crop_centered_on(boundary)
                    elif want_positive:
                        candidate_image, candidate_mask = crop_centered_on(foreground)
                    else:
                        candidate_image, candidate_mask = crop_once()
                    ratio = foreground_ratio(candidate_mask)
                    if ratio > best_ratio:
                        best_image, best_mask, best_ratio = candidate_image, candidate_mask, ratio
                    if (not want_positive and not want_boundary and ratio > 0) or ratio >= self.min_positive_crop_ratio:
                        best_image, best_mask = candidate_image, candidate_mask
                        break
                if best_ratio > foreground_ratio(mask_crop):
                    image_crop, mask_crop = best_image, best_mask
            image, mask = image_crop, mask_crop
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.1:
                image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            image = image.resize((target, target), Image.Resampling.BILINEAR)
            mask = mask.resize((target, target), Image.Resampling.NEAREST)
        return image, mask

    @staticmethod
    def _resize_dct(dct: Tensor, size: tuple[int, int]) -> Tensor:
        width, height = int(size[0]), int(size[1])
        return F.interpolate(dct.unsqueeze(0).float(), size=(height, width), mode="nearest").squeeze(0)

    def _joint_geometry_with_dct(self, image: Image.Image, mask: Image.Image, dct: Tensor) -> tuple[Image.Image, Image.Image, Tensor]:
        target = self.image_size
        if self.training:
            scale = random.uniform(0.75, 1.5)
            shorter = max(1, min(image.size))
            ratio = target * scale / shorter
            new_size = (
                max(target, round(image.width * ratio)),
                max(target, round(image.height * ratio)),
            )
            image = image.resize(new_size, Image.Resampling.BILINEAR)
            mask = mask.resize(new_size, Image.Resampling.NEAREST)
            dct = self._resize_dct(dct, new_size)

            def crop_at(left: int, top: int) -> tuple[Image.Image, Image.Image, Tensor]:
                left = max(0, min(left, image.width - target))
                top = max(0, min(top, image.height - target))
                box = (left, top, left + target, top + target)
                return image.crop(box), mask.crop(box), dct[:, top : top + target, left : left + target]

            def crop_once() -> tuple[Image.Image, Image.Image, Tensor]:
                return crop_at(
                    random.randint(0, image.width - target),
                    random.randint(0, image.height - target),
                )

            def foreground_ratio(candidate: Image.Image) -> float:
                return float((np.asarray(candidate, dtype=np.uint8) > 0).mean())

            def crop_centered_on(points: np.ndarray) -> tuple[Image.Image, Image.Image, Tensor]:
                y, x = points[random.randrange(len(points))]
                jitter = target // 4
                left = int(x) - target // 2 + random.randint(-jitter, jitter)
                top = int(y) - target // 2 + random.randint(-jitter, jitter)
                return crop_at(left, top)

            mask_array = np.asarray(mask, dtype=np.uint8) > 0
            foreground = np.argwhere(mask_array)
            image_crop, mask_crop, dct_crop = crop_once()
            if len(foreground) > 0:
                boundary = foreground
                if self.boundary_crop_probability > 0:
                    padded = np.pad(mask_array, 1, mode="constant", constant_values=False)
                    center = padded[1:-1, 1:-1]
                    eroded = (
                        padded[:-2, 1:-1] & padded[2:, 1:-1]
                        & padded[1:-1, :-2] & padded[1:-1, 2:]
                        & center
                    )
                    boundary_points = np.argwhere(center & ~eroded)
                    if len(boundary_points) > 0:
                        boundary = boundary_points
                want_positive = random.random() < self.positive_crop_probability
                want_boundary = random.random() < self.boundary_crop_probability
                best_image, best_mask, best_dct = image_crop, mask_crop, dct_crop
                best_ratio = foreground_ratio(mask_crop)
                tries = max(1, self.crop_retries)
                for _ in range(tries):
                    if want_boundary:
                        candidate_image, candidate_mask, candidate_dct = crop_centered_on(boundary)
                    elif want_positive:
                        candidate_image, candidate_mask, candidate_dct = crop_centered_on(foreground)
                    else:
                        candidate_image, candidate_mask, candidate_dct = crop_once()
                    ratio = foreground_ratio(candidate_mask)
                    if ratio > best_ratio:
                        best_image, best_mask, best_dct, best_ratio = candidate_image, candidate_mask, candidate_dct, ratio
                    if (not want_positive and not want_boundary and ratio > 0) or ratio >= self.min_positive_crop_ratio:
                        best_image, best_mask, best_dct = candidate_image, candidate_mask, candidate_dct
                        break
                if best_ratio > foreground_ratio(mask_crop):
                    image_crop, mask_crop, dct_crop = best_image, best_mask, best_dct
            image, mask, dct = image_crop, mask_crop, dct_crop
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                dct = torch.flip(dct, dims=(-1,))
            if random.random() < 0.1:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                dct = torch.flip(dct, dims=(-2,))
        else:
            image = image.resize((target, target), Image.Resampling.BILINEAR)
            mask = mask.resize((target, target), Image.Resampling.NEAREST)
            dct = self._resize_dct(dct, (target, target))
        return image, mask, dct.contiguous()

    def _photometric(self, image: Image.Image) -> Image.Image:
        if self.color_jitter > 0:
            amount = self.color_jitter
            image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - amount, 1 + amount))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - amount, 1 + amount))
            image = ImageEnhance.Color(image).enhance(random.uniform(1 - amount, 1 + amount))
        if random.random() < self.blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.3)))
        if random.random() < self.jpeg_probability:
            quality = random.randint(*self.jpeg_quality)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        return image


    def _source_qtable_path(self, record: Record) -> Path | None:
        stem = record.image.stem
        if record.image.suffix.lower() in {".jpg", ".jpeg"}:
            direct = record.image
            if direct.exists():
                return direct
        subdir = "Au" if record.label == 0 else "Tp"
        candidates = []
        if record.image.suffix.lower() in {".jpg", ".jpeg"}:
            candidates.append(self.dct_cat_root / "CASIA 2.0" / subdir / f"{stem}.jpg")
        candidates.append(self.dct_cat_root / "CASIA 2.0" / "jpg" / f"{stem}.jpg")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _read_source_qtable(path: Path | None) -> list[list[int]] | None:
        if path is None or path.suffix.lower() not in {".jpg", ".jpeg"} or not path.exists():
            return None
        try:
            with Image.open(path) as handle:
                quantization = handle.quantization
        except Exception:
            return None
        if not quantization:
            return None
        first_key = sorted(quantization.keys())[0]
        return [list(int(v) for v in quantization[first_key])]

    def _jpeg_info_from_image(self, image: Image.Image, quality: int, qtable_source: Path | None = None) -> tuple[Tensor, Tensor]:
        try:
            import jpegio
        except ImportError as exc:
            raise ImportError("dct_forensic=true requires jpegio") from exc
        rgb = image.convert("RGB")
        width, height = rgb.size
        pad_w = ((width + 7) // 8) * 8
        pad_h = ((height + 7) // 8) * 8
        if (pad_w, pad_h) != (width, height):
            padded = Image.new("RGB", (pad_w, pad_h), color=(128, 128, 128))
            padded.paste(rgb, (0, 0))
            rgb = padded
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                tmp_name = handle.name
            qtables = self._read_source_qtable(qtable_source) if self.dct_qtable_from_source else None
            if qtables is not None:
                rgb.save(tmp_name, format="JPEG", qtables=qtables, subsampling=0)
            else:
                rgb.save(tmp_name, format="JPEG", quality=int(quality), subsampling=0)
            jpeg = jpegio.read(tmp_name)
            coef = jpeg.coef_arrays[0].astype("float32")
            ci = jpeg.comp_info[0]
            qtable = jpeg.quant_tables[ci.quant_tbl_no].astype("float32")
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        coef = coef[:pad_h, :pad_w]
        dct = np.zeros((21, pad_h, pad_w), dtype=np.float32)
        dct[0] = (coef == 0)
        abs_coef = np.abs(coef)
        for value in range(1, 20):
            dct[value] = (abs_coef == value)
        dct[20] = (abs_coef >= 20)
        dct = dct[:, :height, :width]
        return torch.from_numpy(dct), torch.from_numpy(qtable[None, :, :])

    @staticmethod
    def _dct_volume_from_coef(coef: np.ndarray, height: int, width: int) -> Tensor:
        coef = coef[:height, :width].astype("float32", copy=False)
        dct = np.zeros((21, height, width), dtype=np.float32)
        dct[0] = (coef == 0)
        abs_coef = np.abs(coef)
        for value in range(1, 20):
            dct[value] = (abs_coef == value)
        dct[20] = (abs_coef >= 20)
        return torch.from_numpy(dct)

    def _jpeg_info_from_original(self, record: Record, size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        width, height = int(size[0]), int(size[1])
        unavailable = (
            torch.zeros((21, height, width), dtype=torch.float32),
            torch.ones((1, 8, 8), dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )
        source = record.image
        if source.suffix.lower() not in {".jpg", ".jpeg"} or not source.exists():
            return unavailable
        try:
            import jpegio
        except ImportError as exc:
            raise ImportError("dct_forensic=true with dct_source_mode=original_jpeg requires jpegio") from exc
        try:
            jpeg = jpegio.read(str(source))
            coef = jpeg.coef_arrays[0].astype("float32")
            ci = jpeg.comp_info[0]
            qtable = jpeg.quant_tables[ci.quant_tbl_no].astype("float32")
        except Exception:
            return unavailable
        return self._dct_volume_from_coef(coef, height, width), torch.from_numpy(qtable[None, :, :]), torch.tensor([1.0], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image) as handle:
            image = handle.convert("RGB")
        if record.mask is None:
            mask = Image.new("L", image.size, color=0)
        else:
            with Image.open(record.mask) as handle:
                mask = handle.convert("L")
        dct_tensor = qtable_tensor = dct_available = None
        if self.dct_forensic and self.dct_source_mode == "original_jpeg":
            dct_tensor, qtable_tensor, dct_available = self._jpeg_info_from_original(record, image.size)
            image, mask, dct_tensor = self._joint_geometry_with_dct(image, mask, dct_tensor)
        else:
            image, mask = self._joint_geometry(image, mask)
        if self.training:
            image = self._photometric(image)
        if self.dct_forensic and self.dct_source_mode != "original_jpeg":
            dct_tensor, qtable_tensor = self._jpeg_info_from_image(image, self.dct_jpeg_quality, self._source_qtable_path(record))
            dct_available = torch.tensor([1.0], dtype=torch.float32)
        mask_tensor = (pil_to_tensor(mask) > 0.5).float()
        patch_label = float(mask_tensor.any()) if self.training else float(record.label)
        sample = {
            "image": pil_to_tensor(image),
            "mask": mask_tensor,
            "label": torch.tensor([patch_label], dtype=torch.float32),
            "path": str(record.image),
            "dataset": record.dataset,
        }
        if dct_tensor is not None and qtable_tensor is not None:
            sample["dct"] = dct_tensor
            sample["qtable"] = qtable_tensor
            sample["dct_available"] = dct_available if dct_available is not None else torch.tensor([1.0], dtype=torch.float32)
        return sample


def build_loader(config: Mapping[str, Any], split: str, shuffle: bool | None = None) -> DataLoader:
    root = Path(config["root"])
    manifest = _resolve(root, config["manifest"])
    records = read_manifest(manifest, root, split)
    training = split == "train"
    dataset = ManipulationDataset(
        records,
        image_size=config.get("image_size", 512),
        training=training,
        jpeg_probability=config.get("jpeg_probability", 0.0) if training else 0.0,
        jpeg_quality=config.get("jpeg_quality", (45, 96)),
        blur_probability=config.get("blur_probability", 0.0) if training else 0.0,
        color_jitter=config.get("color_jitter", 0.0) if training else 0.0,
        positive_crop_probability=config.get("positive_crop_probability", 0.0) if training else 0.0,
        boundary_crop_probability=config.get("boundary_crop_probability", 0.0) if training else 0.0,
        min_positive_crop_ratio=config.get("min_positive_crop_ratio", 0.0),
        crop_retries=config.get("crop_retries", 10),
        dct_forensic=config.get("dct_forensic", False),
        dct_jpeg_quality=config.get("dct_jpeg_quality", 100),
        dct_qtable_from_source=config.get("dct_qtable_from_source", True),
        dct_cat_root=config.get("dct_cat_root"),
        dct_source_mode=config.get("dct_source_mode", "reencoded"),
    )
    sampler = None
    if training and config.get("authentic_probability", 0.0) > 0:
        desired = float(config["authentic_probability"])
        authentic = max(1, sum(record.label == 0 for record in records))
        manipulated = max(1, len(records) - authentic)
        weights = [desired / authentic if record.label == 0 else (1.0 - desired) / manipulated for record in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(records), replacement=True)
    if shuffle is None:
        shuffle = training and sampler is None
    batch_size = config.get("batch_size", 8)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=config.get("num_workers", 4) > 0,
    )
