"""
Rebuild a leakage-resistant dataset split from existing Dataset/{train,validation,test}.

Strategy:
1) Collect all images across existing splits.
2) Deduplicate using perceptual hash (image-content based).
3) Keep only one sample per (class, hash) group.
4) Split per class into new train/validation/test sets.

Output:
- Dataset_clean/train|validation|test/<class>/*
- diagnosis_results/clean_split_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")


@dataclass
class Sample:
    path: Path
    klass: str
    split: str
    phash: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild clean dataset split")
    parser.add_argument("--src", type=Path, default=Path("Dataset"))
    parser.add_argument("--dst", type=Path, default=Path("Dataset_clean"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--report", type=Path, default=Path("diagnosis_results/clean_split_report.json"))
    parser.add_argument(
        "--hash-mode",
        type=str,
        default="exact",
        choices=["exact", "phash"],
        help="Deduplication hash mode: exact (sha1 bytes) or phash (perceptual)",
    )
    return parser.parse_args()


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((32, 32), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32)
    gray = arr.mean(axis=2)
    pooled = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    bits = pooled > pooled.mean()
    v = 0
    for b in bits.flatten():
        v = (v << 1) | int(b)
    return int(v)


def exact_hash(path: Path) -> int:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return int(h.hexdigest(), 16)


def collect_samples(src: Path, hash_mode: str) -> List[Sample]:
    samples: List[Sample] = []
    for split in ["train", "validation", "test"]:
        for klass in CLASSES:
            cdir = src / split / klass
            if not cdir.exists():
                continue
            paths: List[Path] = []
            for ext in EXTS:
                paths.extend(cdir.glob(ext))
            for p in sorted(paths):
                try:
                    h = exact_hash(p) if hash_mode == "exact" else perceptual_hash(p)
                    samples.append(Sample(path=p, klass=klass, split=split, phash=h))
                except Exception:
                    continue
    return samples


def deduplicate(samples: List[Sample]) -> Tuple[List[Sample], Dict[str, int]]:
    # key by (class, hash) to avoid cross-class collisions in hash space
    buckets: Dict[Tuple[str, int], List[Sample]] = defaultdict(list)
    for s in samples:
        buckets[(s.klass, s.phash)].append(s)

    deduped: List[Sample] = []
    dup_count = 0

    for _, group in buckets.items():
        # Prefer train > validation > test when selecting representative
        group_sorted = sorted(group, key=lambda x: {"train": 0, "validation": 1, "test": 2}[x.split])
        deduped.append(group_sorted[0])
        dup_count += max(0, len(group_sorted) - 1)

    summary = {
        "original_samples": len(samples),
        "deduplicated_samples": len(deduped),
        "removed_as_duplicates": dup_count,
    }
    return deduped, summary


def split_by_class(samples: List[Sample], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, Dict[str, List[Sample]]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.klass].append(s)

    out: Dict[str, Dict[str, List[Sample]]] = {}
    for klass in CLASSES:
        items = by_class.get(klass, [])
        idx = np.arange(len(items))
        rng.shuffle(idx)
        items = [items[i] for i in idx]

        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(n_train, 1), max(1, n - 2)) if n >= 3 else max(1, n)
        n_val = min(max(n_val, 1), max(1, n - n_train - 1)) if n >= 3 else 0
        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 3:
            n_test = 1
            n_train = max(1, n_train - 1)

        out[klass] = {
            "train": items[:n_train],
            "validation": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }
    return out


def copy_split(split_map: Dict[str, Dict[str, List[Sample]]], dst: Path) -> Dict[str, Dict[str, int]]:
    if dst.exists():
        shutil.rmtree(dst)

    counts: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}

    for split in ["train", "validation", "test"]:
        for klass in CLASSES:
            out_dir = dst / split / klass
            out_dir.mkdir(parents=True, exist_ok=True)
            assigned = split_map.get(klass, {}).get(split, [])
            counts[split][klass] = len(assigned)
            for i, s in enumerate(assigned):
                suffix = s.path.suffix.lower()
                out_name = f"{klass}_{i:06d}{suffix}"
                shutil.copy2(s.path, out_dir / out_name)

    return counts


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    samples = collect_samples(args.src, args.hash_mode)
    deduped, dedupe_summary = deduplicate(samples)

    split_map = split_by_class(
        deduped,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    counts = copy_split(split_map, args.dst)

    report = {
        "source": str(args.src),
        "destination": str(args.dst),
        "hash_mode": args.hash_mode,
        "class_names": CLASSES,
        "dedupe_summary": dedupe_summary,
        "split_counts": counts,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("CLEAN DATASET REBUILD COMPLETE")
    print("=" * 80)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
