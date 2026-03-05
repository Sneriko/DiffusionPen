#!/usr/bin/env python3
"""Convert PAGE-XML + page images into line-image manifests for DiffusionPen-style training.

Output schema (TSV):
split\timage_path\txml_path\tline_id\twriter_id\ttranscription\tbbox_xyxy\tpolygon_xy

- split: train|val|test (or custom)
- image_path: path to cropped line image (relative by default)
- xml_path: source PAGE-XML path
- line_id: TextLine/@id
- writer_id: pseudo writer assignment (configurable)
- transcription: UnicodeText string
- bbox_xyxy: "x1,y1,x2,y2"
- polygon_xy: "x1,y1 x2,y2 ..."
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


def find_with_ns(root: ET.Element, tag: str):
    """Yield elements ending with tag irrespective of namespace prefix."""
    for elem in root.iter():
        if elem.tag.endswith(tag):
            yield elem


def parse_points(points_str: str):
    pts = []
    for token in points_str.strip().split():
        x_str, y_str = token.split(",")
        pts.append((int(float(x_str)), int(float(y_str))))
    return pts


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def clamp_bbox(x1, y1, x2, y2, w, h, pad=2):
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def xml_image_filename(root: ET.Element):
    page_nodes = list(find_with_ns(root, "Page"))
    if not page_nodes:
        return None
    return page_nodes[0].attrib.get("imageFilename")


def extract_lines(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for tl in find_with_ns(root, "TextLine"):
        line_id = tl.attrib.get("id", "")

        coords = None
        for c in tl:
            if c.tag.endswith("Coords"):
                coords = c
                break
        if coords is None:
            continue

        pts_attr = coords.attrib.get("points", "").strip()
        if not pts_attr:
            continue

        points = parse_points(pts_attr)

        text = ""
        for u in find_with_ns(tl, "Unicode"):
            text = (u.text or "").strip()
            if text:
                break

        rows.append({"line_id": line_id, "points": points, "transcription": text})

    return xml_image_filename(root), rows


def default_volume_id(xml_path: Path, pages_root: Path) -> str:
    rel = xml_path.relative_to(pages_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return rel.parent.name or "volume0"


def default_page_id(xml_path: Path) -> str:
    return xml_path.stem


def split_for_volume(volume_id: str, vol_to_split: dict[str, str], train_ratio: float, val_ratio: float):
    if volume_id in vol_to_split:
        return vol_to_split[volume_id]
    r = random.random()
    if r < train_ratio:
        s = "train"
    elif r < train_ratio + val_ratio:
        s = "val"
    else:
        s = "test"
    vol_to_split[volume_id] = s
    return s


def assign_writer_id(mode: str, volume_id: str, page_id: str):
    if mode == "volume":
        return f"v::{volume_id}"
    if mode == "page":
        return f"p::{page_id}"
    if mode == "volume_page":
        return f"vp::{volume_id}::{page_id}"
    raise ValueError(f"Unknown writer mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description="Build line-crop manifest from PAGE-XML")
    ap.add_argument("--xml-root", type=Path, required=True, help="Root containing PAGE-XML files")
    ap.add_argument("--images-root", type=Path, required=True, help="Root containing full page images")
    ap.add_argument("--out-root", type=Path, required=True, help="Root to write line crops + manifest")
    ap.add_argument("--manifest-name", default="lines_manifest.tsv")
    ap.add_argument("--image-exts", default=".jpg,.jpeg,.png,.tif,.tiff")
    ap.add_argument("--writer-mode", choices=["volume", "page", "volume_page"], default="volume")
    ap.add_argument("--split-by", choices=["volume", "page"], default="volume")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-text-len", type=int, default=1)
    ap.add_argument("--crop-pad", type=int, default=2)
    ap.add_argument("--relative-paths", action="store_true", help="Store relative paths in manifest")

    args = ap.parse_args()
    random.seed(args.seed)

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    args.out_root.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_root / "line_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.image_exts.split(",") if e.strip()}

    xml_files = sorted(args.xml_root.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found under {args.xml_root}")

    vol_to_split: dict[str, str] = {}
    writer_vocab: dict[str, int] = {}
    writer_counts = defaultdict(int)
    rows_out = []

    for xml_path in xml_files:
        image_filename, lines = extract_lines(xml_path)
        if not lines:
            continue

        rel_xml = xml_path.relative_to(args.xml_root)

        candidate_paths = []
        if image_filename:
            p = (args.images_root / rel_xml.parent / image_filename)
            candidate_paths.append(p)
            candidate_paths.append(args.images_root / image_filename)

        if not candidate_paths:
            candidate_paths.append(args.images_root / rel_xml.with_suffix(".jpg"))

        page_image = None
        page_path = None
        for cand in candidate_paths:
            if cand.suffix.lower() in exts and cand.exists():
                page_path = cand
                page_image = Image.open(cand).convert("RGB")
                break
            if cand.exists() and cand.suffix:
                page_path = cand
                page_image = Image.open(cand).convert("RGB")
                break

        if page_image is None:
            stem = rel_xml.with_suffix("")
            found = None
            for ext in exts:
                cand = args.images_root / f"{stem}{ext}"
                if cand.exists():
                    found = cand
                    break
            if found is None:
                print(f"[WARN] Missing page image for XML: {xml_path}")
                continue
            page_path = found
            page_image = Image.open(found).convert("RGB")

        w, h = page_image.size
        volume_id = default_volume_id(xml_path, args.xml_root)
        page_id = default_page_id(xml_path)

        if args.split_by == "volume":
            split = split_for_volume(volume_id, vol_to_split, args.train_ratio, args.val_ratio)
        else:
            split = split_for_volume(page_id, vol_to_split, args.train_ratio, args.val_ratio)

        writer_key = assign_writer_id(args.writer_mode, volume_id, page_id)
        if writer_key not in writer_vocab:
            writer_vocab[writer_key] = len(writer_vocab)
        writer_id = writer_vocab[writer_key]

        for li, line in enumerate(lines):
            txt = line["transcription"].strip()
            if len(txt) < args.min_text_len:
                continue

            x1, y1, x2, y2 = bbox_from_points(line["points"])
            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h, pad=args.crop_pad)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = page_image.crop((x1, y1, x2, y2))
            line_id = line["line_id"] or f"line_{li:05d}"

            rel_crop = Path("line_crops") / split / volume_id / f"{page_id}__{line_id}.png"
            abs_crop = args.out_root / rel_crop
            abs_crop.parent.mkdir(parents=True, exist_ok=True)
            crop.save(abs_crop)

            poly = " ".join(f"{x},{y}" for x, y in line["points"])
            bbox = f"{x1},{y1},{x2},{y2}"

            image_path_field = rel_crop.as_posix() if args.relative_paths else str(abs_crop)
            xml_path_field = rel_xml.as_posix() if args.relative_paths else str(xml_path)

            rows_out.append(
                {
                    "split": split,
                    "image_path": image_path_field,
                    "xml_path": xml_path_field,
                    "line_id": line_id,
                    "writer_id": writer_id,
                    "transcription": txt,
                    "bbox_xyxy": bbox,
                    "polygon_xy": poly,
                }
            )
            writer_counts[writer_id] += 1

    manifest_path = args.out_root / args.manifest_name
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_path",
                "xml_path",
                "line_id",
                "writer_id",
                "transcription",
                "bbox_xyxy",
                "polygon_xy",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"Wrote {len(rows_out)} lines -> {manifest_path}")
    print(f"Pseudo-writer classes: {len(writer_vocab)}")
    if writer_counts:
        sizes = sorted(writer_counts.values())
        print(
            "Writer lines stats: "
            f"min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}"
        )


if __name__ == "__main__":
    main()
