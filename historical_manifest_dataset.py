from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def resize_keep_aspect(img: Image.Image, target_h: int, target_w: int, pad_color=255) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=(pad_color, pad_color, pad_color))
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(img, (x, y))
    return canvas


@dataclass
class ManifestRow:
    split: str
    image_path: str
    xml_path: str
    line_id: str
    writer_id: str
    transcription: str
    bbox_xyxy: str
    polygon_xy: str


class ManifestIndex:
    def __init__(self, rows: Sequence[ManifestRow]):
        self.rows = list(rows)
        self.writer_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            self.writer_to_indices[str(row.writer_id)].append(i)
        self.writers = sorted(self.writer_to_indices.keys())
        self.writer_to_label = {w: i for i, w in enumerate(self.writers)}
        self.label_to_writer = {i: w for w, i in self.writer_to_label.items()}

    @property
    def num_writers(self) -> int:
        return len(self.writers)


def read_manifest(manifest_path: str | Path, split: Optional[str] = None) -> List[ManifestRow]:
    manifest_path = Path(manifest_path)
    rows: List[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"split", "image_path", "xml_path", "line_id", "writer_id", "transcription", "bbox_xyxy", "polygon_xy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        for r in reader:
            if split is not None and r["split"] != split:
                continue
            rows.append(
                ManifestRow(
                    split=r["split"],
                    image_path=r["image_path"],
                    xml_path=r["xml_path"],
                    line_id=r["line_id"],
                    writer_id=r["writer_id"],
                    transcription=r["transcription"],
                    bbox_xyxy=r["bbox_xyxy"],
                    polygon_xy=r["polygon_xy"],
                )
            )
    if not rows:
        raise ValueError(f"No rows found in manifest for split={split!r}")
    return rows


class BaseManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_height: int = 96,
        image_width: int = 768,
        grayscale: bool = False,
        writer_to_label: Optional[Dict[str, int]] = None,
        style_refs: int = 5,
        min_text_len: int = 1,
    ):
        all_rows = read_manifest(manifest_path, split=split)
        rows = [r for r in all_rows if len(r.transcription.strip()) >= min_text_len]
        if not rows:
            raise ValueError(f"No usable rows for split={split!r}")
        self.rows = rows
        self.index = ManifestIndex(rows)
        self.image_height = image_height
        self.image_width = image_width
        self.grayscale = grayscale
        self.style_refs = style_refs
        self.split = split

        if writer_to_label is None:
            self.writer_to_label = self.index.writer_to_label
        else:
            self.writer_to_label = dict(writer_to_label)
            unknown = sorted({r.writer_id for r in self.rows if r.writer_id not in self.writer_to_label})
            if unknown:
                raise ValueError(f"Split {split!r} contains writer_ids not seen in training: {unknown[:10]}")

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.writer_to_indices = defaultdict(list)
        for i, r in enumerate(self.rows):
            self.writer_to_indices[r.writer_id].append(i)
        self.writer_ids = sorted(self.writer_to_indices.keys())

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert("L")
            img = Image.merge("RGB", (img, img, img))
        else:
            img = img.convert("RGB")
        return resize_keep_aspect(img, self.image_height, self.image_width)

    def _sample_positive_index(self, idx: int) -> int:
        row = self.rows[idx]
        candidates = self.writer_to_indices[row.writer_id]
        if len(candidates) == 1:
            return idx
        j = idx
        while j == idx:
            j = random.choice(candidates)
        return j

    def _sample_negative_index(self, idx: int) -> int:
        row = self.rows[idx]
        neg_writer = random.choice([w for w in self.writer_ids if w != row.writer_id])
        return random.choice(self.writer_to_indices[neg_writer])

    def _sample_style_indices(self, idx: int, k: int) -> List[int]:
        row = self.rows[idx]
        candidates = self.writer_to_indices[row.writer_id]
        if len(candidates) >= k:
            return random.sample(candidates, k)
        return random.choices(candidates, k=k)


class StyleEncoderManifestDataset(BaseManifestDataset):
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pos_idx = self._sample_positive_index(idx)
        neg_idx = self._sample_negative_index(idx)

        anchor = self.transform(self._load_image(row.image_path))
        positive = self.transform(self._load_image(self.rows[pos_idx].image_path))
        negative = self.transform(self._load_image(self.rows[neg_idx].image_path))

        writer_label = self.writer_to_label[row.writer_id]
        return {
            "image": anchor,
            "positive": positive,
            "negative": negative,
            "writer_label": torch.tensor(writer_label, dtype=torch.long),
            "writer_id": row.writer_id,
            "transcription": row.transcription,
            "image_path": row.image_path,
        }


class Stage2ManifestDataset(BaseManifestDataset):
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        style_indices = self._sample_style_indices(idx, self.style_refs)

        image = self.transform(self._load_image(row.image_path))
        style_images = torch.stack([self.transform(self._load_image(self.rows[j].image_path)) for j in style_indices], dim=0)
        writer_label = self.writer_to_label[row.writer_id]

        return {
            "image": image,
            "transcription": row.transcription,
            "writer_label": torch.tensor(writer_label, dtype=torch.long),
            "writer_id": row.writer_id,
            "style_images": style_images,
            "image_path": row.image_path,
        }


def make_train_val_datasets(
    manifest_path: str | Path,
    image_height: int = 96,
    image_width: int = 768,
    grayscale: bool = False,
    style_refs: int = 5,
):
    train_ds = Stage2ManifestDataset(
        manifest_path=manifest_path,
        split="train",
        image_height=image_height,
        image_width=image_width,
        grayscale=grayscale,
        style_refs=style_refs,
        writer_to_label=None,
    )
    val_ds = Stage2ManifestDataset(
        manifest_path=manifest_path,
        split="val",
        image_height=image_height,
        image_width=image_width,
        grayscale=grayscale,
        style_refs=style_refs,
        writer_to_label=train_ds.writer_to_label,
    )
    return train_ds, val_ds
