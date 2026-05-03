"""
Comprehensive baseline diagnosis for rice disease classifier.

Outputs:
- Split metrics: train/validation/test (accuracy, precision, recall, F1)
- Confusion matrices
- Calibration metrics: NLL, Brier score, ECE
- Overfitting gap summary
- Dataset integrity checks: class imbalance + duplicate leakage across splits
- OOD behavior on optional OOD folder and synthetic random images
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    log_loss,
)


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


@dataclass
class SplitMetrics:
    name: str
    samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_confidence: float
    mean_correct_confidence: float
    mean_incorrect_confidence: float
    nll: float
    ece_15_bins: float


@dataclass
class OODMetrics:
    source: str
    samples: int
    mean_max_confidence: float
    p95_max_confidence: float
    fraction_above_0_8: float
    fraction_above_0_9: float
    mean_entropy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose overfitting and reliability")
    parser.add_argument("--model", type=Path, default=Path("models/rice_disease_model_final.keras"))
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--ood-dir", type=Path, default=Path("OOD"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnosis_results"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset(split_dir: Path, shuffle: bool) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def collect_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    probs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        probs_list.append(p)
        labels_list.append(y.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return labels.astype(np.int64), probs.astype(np.float64)


def ece_score(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def split_metrics(name: str, y_true: np.ndarray, probs: np.ndarray) -> SplitMetrics:
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        average="macro",
        zero_division=0,
    )

    correct = y_pred == y_true
    nll = log_loss(y_true, probs, labels=np.arange(len(CLASSES)))

    return SplitMetrics(
        name=name,
        samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(p),
        macro_recall=float(r),
        macro_f1=float(f1),
        mean_confidence=float(conf.mean()),
        mean_correct_confidence=float(conf[correct].mean() if np.any(correct) else 0.0),
        mean_incorrect_confidence=float(conf[~correct].mean() if np.any(~correct) else 0.0),
        nll=float(nll),
        ece_15_bins=ece_score(probs, y_true, bins=15),
    )


def simple_hash(path: Path) -> int:
    img = tf.keras.utils.load_img(path, target_size=(32, 32))
    arr = tf.keras.utils.img_to_array(img).astype(np.float32)
    gray = arr.mean(axis=2)
    pooled = gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    bits = pooled > pooled.mean()
    value = 0
    for b in bits.flatten():
        value = (value << 1) | int(b)
    return value


def gather_image_paths(split_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for c in CLASSES:
        class_dir = split_dir / c
        if not class_dir.exists():
            continue
        paths.extend(sorted(class_dir.glob("*.jpg")))
        paths.extend(sorted(class_dir.glob("*.jpeg")))
        paths.extend(sorted(class_dir.glob("*.png")))
        paths.extend(sorted(class_dir.glob("*.webp")))
    return paths


def duplicate_leakage_report(data_root: Path) -> Dict[str, int]:
    split_names = ["train", "validation", "test"]
    split_hashes: Dict[str, set[int]] = {}
    for s in split_names:
        hset: set[int] = set()
        for p in gather_image_paths(data_root / s):
            try:
                hset.add(simple_hash(p))
            except Exception:
                continue
        split_hashes[s] = hset

    return {
        "train_validation_exact_hash_overlap": int(len(split_hashes["train"] & split_hashes["validation"])),
        "train_test_exact_hash_overlap": int(len(split_hashes["train"] & split_hashes["test"])),
        "validation_test_exact_hash_overlap": int(len(split_hashes["validation"] & split_hashes["test"])),
    }


def class_distribution(data_root: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in ["train", "validation", "test"]:
        out[split] = {}
        for c in CLASSES:
            cdir = data_root / split / c
            n = 0
            if cdir.exists():
                n = sum(1 for _ in cdir.glob("*.*"))
            out[split][c] = int(n)
    return out


def synthetic_ood_batch(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        if i % 3 == 0:
            arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        elif i % 3 == 1:
            v = int(rng.integers(0, 256))
            arr = np.full((224, 224, 3), v, dtype=np.uint8)
        else:
            x = np.linspace(0, 255, 224, dtype=np.float32)
            y = np.linspace(0, 255, 224, dtype=np.float32)
            xv, yv = np.meshgrid(x, y)
            arr = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
        samples.append(arr)
    return np.stack(samples, axis=0).astype(np.float32)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return h / math.log(probs.shape[1])


def ood_metrics_from_probs(source: str, probs: np.ndarray) -> OODMetrics:
    conf = probs.max(axis=1)
    ent = entropy_from_probs(probs)
    return OODMetrics(
        source=source,
        samples=int(len(probs)),
        mean_max_confidence=float(conf.mean()),
        p95_max_confidence=float(np.percentile(conf, 95)),
        fraction_above_0_8=float(np.mean(conf >= 0.8)),
        fraction_above_0_9=float(np.mean(conf >= 0.9)),
        mean_entropy=float(ent.mean()),
    )


def evaluate_ood_folder(model: tf.keras.Model, ood_dir: Path) -> OODMetrics | None:
    if not ood_dir.exists():
        return None
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        paths.extend(ood_dir.rglob(ext))
    if not paths:
        return None

    imgs = []
    for p in paths:
        try:
            img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img)
            imgs.append(arr)
        except Exception:
            continue
    if not imgs:
        return None

    x = np.stack(imgs, axis=0).astype(np.float32)
    probs = model.predict(x, verbose=0)
    return ood_metrics_from_probs("ood_folder", probs)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)

    train_ds = build_dataset(args.data_root / "train", shuffle=False)
    val_ds = build_dataset(args.data_root / "validation", shuffle=False)
    test_ds = build_dataset(args.data_root / "test", shuffle=False)

    y_train, p_train = collect_predictions(model, train_ds)
    y_val, p_val = collect_predictions(model, val_ds)
    y_test, p_test = collect_predictions(model, test_ds)

    m_train = split_metrics("train", y_train, p_train)
    m_val = split_metrics("validation", y_val, p_val)
    m_test = split_metrics("test", y_test, p_test)

    cm_val = confusion_matrix(y_val, p_val.argmax(axis=1), labels=np.arange(len(CLASSES)))
    cm_test = confusion_matrix(y_test, p_test.argmax(axis=1), labels=np.arange(len(CLASSES)))

    overfit = {
        "train_minus_val_accuracy_gap": float(m_train.accuracy - m_val.accuracy),
        "train_minus_test_accuracy_gap": float(m_train.accuracy - m_test.accuracy),
        "val_minus_test_accuracy_gap": float(m_val.accuracy - m_test.accuracy),
    }

    leakage = duplicate_leakage_report(args.data_root)
    distribution = class_distribution(args.data_root)

    synth = synthetic_ood_batch(300, args.seed)
    p_synth = model.predict(synth, verbose=0)
    ood_reports: List[OODMetrics] = [ood_metrics_from_probs("synthetic_ood", p_synth)]

    real_ood = evaluate_ood_folder(model, args.ood_dir)
    if real_ood is not None:
        ood_reports.append(real_ood)

    report = {
        "model": str(args.model),
        "classes": CLASSES,
        "split_metrics": {
            "train": asdict(m_train),
            "validation": asdict(m_val),
            "test": asdict(m_test),
        },
        "overfitting_gaps": overfit,
        "dataset_distribution": distribution,
        "duplicate_leakage_hash_overlaps": leakage,
        "ood": [asdict(x) for x in ood_reports],
        "confusion_matrix_validation": cm_val.tolist(),
        "confusion_matrix_test": cm_test.tolist(),
        "classification_report_validation": classification_report(
            y_val,
            p_val.argmax(axis=1),
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_test": classification_report(
            y_test,
            p_test.argmax(axis=1),
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
    }

    out_json = args.output_dir / "baseline_diagnosis.json"
    write_json(out_json, report)

    print("=" * 80)
    print("BASELINE DIAGNOSIS COMPLETE")
    print("=" * 80)
    print(f"Train acc: {m_train.accuracy:.4f} | Val acc: {m_val.accuracy:.4f} | Test acc: {m_test.accuracy:.4f}")
    print(f"Train-Val gap: {overfit['train_minus_val_accuracy_gap']:.4f}")
    print(f"Val ECE: {m_val.ece_15_bins:.4f} | Test ECE: {m_test.ece_15_bins:.4f}")
    for item in ood_reports:
        print(
            f"OOD[{item.source}] n={item.samples} "
            f"mean_conf={item.mean_max_confidence:.4f} p95={item.p95_max_confidence:.4f} "
            f"frac>=0.9={item.fraction_above_0_9:.4f}"
        )
    print(f"Saved report: {out_json}")


if __name__ == "__main__":
    main()
