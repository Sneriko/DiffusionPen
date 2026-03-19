#!/usr/bin/env python3
"""Convert PAGE-XML + page images into line-image manifests for DiffusionPen-style training.

Output schema (TSV):
split\timage_path\txml_path\tline_id\twriter_id\ttranscription\tbbox_xyxy\tpolygon_xy

This version:
- finds PAGE XML recursively under any folder path containing "page"
- ignores ALTO XML
- matches images by PAGE imageFilename or basename
- derives volume_id from filename stem before first underscore
- crops by polygon, not simple bbox
- masks out everything outside the polygon
- saves RGBA PNGs with transparent background by default
- creates only train/val splits
- assigns splits deterministically by key count, not per-file random draw
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def find_with_ns(root: ET.Element, tag: str):
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


def shift_points(points, dx, dy):
    return [(x - dx, y - dy) for x, y in points]


def polygon_crop_with_mask(page_image: Image.Image, points, pad=2, background="transparent"):
    """Crop tight bbox around polygon and mask out everything outside polygon."""
    w, h = page_image.size
    x1, y1, x2, y2 = bbox_from_points(points)
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h, pad=pad)

    crop_box = (x1, y1, x2, y2)
    crop = page_image.crop(crop_box)

    shifted = shift_points(points, x1, y1)

    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(shifted, fill=255)

    if background == "transparent":
        rgba = crop.convert("RGBA")
        rgba.putalpha(mask)
        return rgba, (x1, y1, x2, y2), shifted

    white_bg = Image.new("RGB", crop.size, (255, 255, 255))
    rgb_crop = crop.convert("RGB")
    composited = Image.composite(rgb_crop, white_bg, mask)
    return composited, (x1, y1, x2, y2), shifted


def xml_image_filename(root: ET.Element):
    page_nodes = list(find_with_ns(root, "Page"))
    if not page_nodes:
        return None
    return page_nodes[0].attrib.get("imageFilename")


def is_pagexml(root: ET.Element) -> bool:
    return root.tag.endswith("PcGts")


def extract_lines(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if not is_pagexml(root):
        return None, []

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

        try:
            points = parse_points(pts_attr)
        except Exception:
            continue

        text = ""
        for u in find_with_ns(tl, "Unicode"):
            text = (u.text or "").strip()
            if text:
                break

        rows.append({"line_id": line_id, "points": points, "transcription": text})

    return xml_image_filename(root), rows


def volume_id_from_stem(path: Path) -> str:
    """Infer volume id from basename: <volume>_<page> -> <volume>."""
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def default_page_id(xml_path: Path) -> str:
    return xml_path.stem


def make_split_map(keys, train_ratio: float, seed: int):
    """Create a deterministic train/val split map from unique keys.

    Only train and val are used. If there are at least 2 unique keys,
    guarantees at least 1 key in val and at least 1 in train.
    """
    unique_keys = sorted(set(keys))
    if not unique_keys:
        raise ValueError("No split keys found.")

    rng = random.Random(seed)
    rng.shuffle(unique_keys)

    n = len(unique_keys)

    if n == 1:
        return {unique_keys[0]: "train"}

    n_train = int(round(n * train_ratio))
    n_train = max(1, min(n - 1, n_train))
    n_val = n - n_train

    if n_val < 1:
        n_val = 1
        n_train = n - 1

    split_map = {}
    for i, key in enumerate(unique_keys):
        if i < n_train:
            split_map[key] = "train"
        else:
            split_map[key] = "val"

    return split_map


def assign_writer_id(mode: str, volume_id: str, page_id: str):
    if mode == "volume":
        return f"v::{volume_id}"
    if mode == "page":
        return f"p::{page_id}"
    if mode == "volume_page":
        return f"vp::{volume_id}::{page_id}"
    raise ValueError(f"Unknown writer mode: {mode}")


def build_image_basename_index(data_root: Path, image_exts: set[str]):
    by_base = defaultdict(list)
    for ext in image_exts:
        for p in data_root.rglob(f"*{ext}"):
            by_base[p.stem].append(p)
    return by_base


def choose_image_for_xml_in_single_root(
    xml_path: Path,
    xml_image_name: str | None,
    basename_index: dict[str, list[Path]],
):
    candidates = []

    if xml_image_name:
        image_base = Path(xml_image_name).stem
        candidates.extend(basename_index.get(image_base, []))

    if not candidates:
        candidates.extend(basename_index.get(xml_path.stem, []))

    if not candidates:
        return None

    filtered = []
    for c in candidates:
        lowered_parts = {part.lower() for part in c.parts}
        if "page" in lowered_parts or "alto" in lowered_parts:
            continue
        filtered.append(c)

    if filtered:
        candidates = filtered

    same_dir = [c for c in candidates if c.parent == xml_path.parent]
    if same_dir:
        return sorted(same_dir)[0]

    rel_parts = xml_path.parts
    scored = []
    for c in candidates:
        c_parts = c.parts
        common = 0
        for a, b in zip(rel_parts, c_parts):
            if a == b:
                common += 1
            else:
                break
        scored.append((common, -len(c_parts), str(c), c))

    scored.sort(reverse=True)
    return scored[0][3]


def resolve_paths(args):
    if args.data_root is not None:
        xml_root = args.data_root
        images_root = args.data_root
    else:
        if args.xml_root is None or args.images_root is None:
            raise ValueError("Use either --data-root OR both --xml-root and --images-root")
        xml_root = args.xml_root
        images_root = args.images_root
    return xml_root, images_root


def find_page_xml_files(root: Path):
    xml_files = []
    for p in root.rglob("*.xml"):
        parent_parts_lower = [part.lower() for part in p.parent.parts]
        if "page" in parent_parts_lower:
            xml_files.append(p)
    return sorted(xml_files)


def find_page_ancestor(path: Path, stop_root: Path) -> Path:
    current = path.parent.resolve()
    stop_root = stop_root.resolve()
    while True:
        if current.name.lower() == "page":
            return current
        if current == stop_root or current.parent == current:
            return stop_root
        current = current.parent


def main():
    ap = argparse.ArgumentParser(description="Build polygon-masked line-crop manifest from PAGE-XML")
    ap.add_argument("--data-root", type=Path, default=None, help="Single root containing both XML and image files (recursive)")
    ap.add_argument("--xml-root", type=Path, default=None, help="Root containing PAGE-XML files")
    ap.add_argument("--images-root", type=Path, default=None, help="Root containing full page images")
    ap.add_argument("--out-root", type=Path, required=True, help="Root to write line crops + manifest")
    ap.add_argument("--manifest-name", default="lines_manifest.tsv")
    ap.add_argument("--image-exts", default=".jpg,.jpeg,.png,.tif,.tiff")
    ap.add_argument("--writer-mode", choices=["volume", "page", "volume_page"], default="volume")
    ap.add_argument("--split-by", choices=["volume", "page"], default="volume")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-text-len", type=int, default=1)
    ap.add_argument("--crop-pad", type=int, default=2)
    ap.add_argument("--relative-paths", action="store_true", help="Store relative paths in manifest")
    ap.add_argument("--background", choices=["transparent", "white"], default="transparent")

    args = ap.parse_args()
    random.seed(args.seed)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train_ratio must be > 0.0 and < 1.0")

    xml_root, images_root = resolve_paths(args)

    args.out_root.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_root / "line_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.image_exts.split(",") if e.strip()}

    if args.data_root is not None:
        xml_files = find_page_xml_files(xml_root)
    else:
        xml_files = sorted(xml_root.rglob("*.xml"))

    if not xml_files:
        raise ValueError(f"No PAGE XML files found under {xml_root}")

    basename_index = None
    if args.data_root is not None:
        basename_index = build_image_basename_index(args.data_root, exts)

    # Build deterministic split map from all unique split keys
    all_split_keys = []
    for xml_path in xml_files:
        volume_id = volume_id_from_stem(xml_path)
        page_id = default_page_id(xml_path)
        split_key = volume_id if args.split_by == "volume" else page_id
        all_split_keys.append(split_key)

    split_map = make_split_map(all_split_keys, args.train_ratio, args.seed)

    writer_vocab: dict[str, int] = {}
    writer_counts = defaultdict(int)
    rows_out = []

    n_parse_errors = 0
    n_missing_images = 0
    n_nonpage_xml = 0

    for xml_path in xml_files:
        try:
            image_filename, lines = extract_lines(xml_path)
        except ET.ParseError as e:
            print(f"[WARN] Skipping invalid XML: {xml_path} ({e})")
            n_parse_errors += 1
            continue
        except Exception as e:
            print(f"[WARN] Failed reading XML: {xml_path} ({e})")
            n_parse_errors += 1
            continue

        if image_filename is None and not lines:
            n_nonpage_xml += 1
            continue

        if not lines:
            continue

        if args.data_root is not None:
            pages_root = find_page_ancestor(xml_path, xml_root.resolve())
        else:
            pages_root = xml_root

        try:
            rel_xml = xml_path.relative_to(pages_root)
        except ValueError:
            rel_xml = Path(xml_path.name)

        page_image = None

        if args.data_root is not None:
            resolved = choose_image_for_xml_in_single_root(xml_path, image_filename, basename_index)
            if resolved is not None and resolved.exists():
                try:
                    page_image = Image.open(resolved).convert("RGB")
                except Exception as e:
                    print(f"[WARN] Could not open image {resolved} for XML {xml_path}: {e}")
                    page_image = None
        else:
            candidate_paths = []
            if image_filename:
                candidate_paths.append(images_root / rel_xml.parent / image_filename)
                candidate_paths.append(images_root / image_filename)

            if not candidate_paths:
                candidate_paths.append(images_root / rel_xml.with_suffix(".jpg"))

            for cand in candidate_paths:
                if cand.exists() and cand.suffix.lower() in exts:
                    try:
                        page_image = Image.open(cand).convert("RGB")
                        break
                    except Exception as e:
                        print(f"[WARN] Could not open image {cand}: {e}")

            if page_image is None:
                stem = rel_xml.with_suffix("")
                for ext in exts:
                    cand = images_root / f"{stem}{ext}"
                    if cand.exists():
                        try:
                            page_image = Image.open(cand).convert("RGB")
                            break
                        except Exception as e:
                            print(f"[WARN] Could not open image {cand}: {e}")

        if page_image is None:
            print(f"[WARN] Missing page image for XML: {xml_path}")
            n_missing_images += 1
            continue

        volume_id = volume_id_from_stem(xml_path)
        page_id = default_page_id(xml_path)

        split_key = volume_id if args.split_by == "volume" else page_id
        split = split_map[split_key]

        writer_key = assign_writer_id(args.writer_mode, volume_id, page_id)
        if writer_key not in writer_vocab:
            writer_vocab[writer_key] = len(writer_vocab)
        writer_id = writer_vocab[writer_key]

        for li, line in enumerate(lines):
            txt = line["transcription"].strip()
            if len(txt) < args.min_text_len:
                continue

            points = line["points"]
            if len(points) < 3:
                continue

            try:
                crop, bbox_xyxy, shifted_polygon = polygon_crop_with_mask(
                    page_image,
                    points,
                    pad=args.crop_pad,
                    background=args.background,
                )
            except Exception as e:
                print(f"[WARN] Failed polygon crop for {xml_path} line {line.get('line_id', li)}: {e}")
                continue

            x1, y1, x2, y2 = bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue

            line_id = line["line_id"] or f"line_{li:05d}"

            rel_crop = Path("line_crops") / split / volume_id / f"{page_id}__{line_id}.png"
            abs_crop = args.out_root / rel_crop
            abs_crop.parent.mkdir(parents=True, exist_ok=True)
            crop.save(abs_crop)

            poly_global = " ".join(f"{x},{y}" for x, y in points)
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
                    "polygon_xy": poly_global,
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
        print(f"Writer lines stats: min={sizes[0]}, median={sizes[len(sizes)//2]}, max={sizes[-1]}")

    split_counts = defaultdict(int)
    for row in rows_out:
        split_counts[row["split"]] += 1

    print("Split counts:")
    print(f"  train: {split_counts['train']}")
    print(f"  val:   {split_counts['val']}")

    print(f"Skipped invalid XML files: {n_parse_errors}")
    print(f"Skipped non-PAGE XML files: {n_nonpage_xml}")
    print(f"Skipped XMLs with missing images: {n_missing_images}")


if __name__ == "__main__":
    main()