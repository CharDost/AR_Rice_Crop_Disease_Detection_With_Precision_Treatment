"""
Recalibrate safeguard thresholds and export coverage/accuracy tradeoff curves.

This script complements calibrate_and_validate.py by producing a wider threshold
surface so you can choose profiles for deployment (conservative/balanced/permissive).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalibrate safeguards and export tradeoffs")
    parser.add_argument("--model", type=Path, default=Path("models/rice_disease_model_rebuild_v2.keras"))
    parser.add_argument("--data-root", type=Path, default=Path("Dataset_clean"))
    parser.add_argument("--ood-dir", type=Path, default=Path("OOD"))
    parser.add_argument("--output", type=Path, default=Path("diagnosis_results/safeguard_tradeoffs_rebuild_v2.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(data_root: Path, split: str) -> tf.data.Dataset:
    ds = tf.keras.utils.image_dataset_from_directory(
        data_root / split,
        labels="inferred",
        class_names=CLASSES,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def collect(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    ys, probs = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        ys.append(y.numpy())
        probs.append(p)
    return np.concatenate(ys, axis=0), np.concatenate(probs, axis=0)


def apply_temperature_to_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    logits_like = np.log(p)
    scaled = logits_like / max(1e-6, temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.6, 3.2, 131):
        p_t = apply_temperature_to_probs(p_val, float(t))
        nll = log_loss(y_val, p_t, labels=np.arange(len(CLASSES)))
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(t)
    return best_t, best_nll


def entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    x = np.clip(p, eps, 1.0)
    h = -np.sum(x * np.log(x), axis=1)
    return h / math.log(p.shape[1])


def accepted_metrics(y_true: np.ndarray, probs: np.ndarray, conf_thr: float, ent_thr: float) -> Dict[str, float]:
    pmax = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ent = entropy(probs)
    accept = (pmax >= conf_thr) & (ent <= ent_thr)

    coverage = float(np.mean(accept))
    if accept.sum() == 0:
        return {
            "coverage": coverage,
            "accepted_accuracy": 0.0,
            "accepted_macro_precision": 0.0,
            "accepted_macro_recall": 0.0,
            "accepted_macro_f1": 0.0,
        }

    yt = y_true[accept]
    yp = pred[accept]
    acc = float(accuracy_score(yt, yp))
    p, r, f1, _ = precision_recall_fscore_support(
        yt,
        yp,
        labels=np.arange(len(CLASSES)),
        average="macro",
        zero_division=0,
    )
    return {
        "coverage": coverage,
        "accepted_accuracy": acc,
        "accepted_macro_precision": float(p),
        "accepted_macro_recall": float(r),
        "accepted_macro_f1": float(f1),
    }


def load_ood_images(ood_dir: Path) -> np.ndarray | None:
    if not ood_dir.exists():
        return None
    paths: List[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        paths.extend(ood_dir.rglob(ext))
    imgs = []
    for p in paths:
        try:
            img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
            imgs.append(tf.keras.utils.img_to_array(img))
        except Exception:
            continue
    if not imgs:
        return None
    return np.stack(imgs, axis=0).astype(np.float32)


def synthetic_ood(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arrs: List[np.ndarray] = []
    for i in range(n):
        if i % 2 == 0:
            arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        else:
            val = int(rng.integers(0, 256))
            arr = np.full((224, 224, 3), val, dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs, axis=0).astype(np.float32)


def ood_acceptance_rate(probs: np.ndarray, conf_thr: float, ent_thr: float) -> float:
    pmax = probs.max(axis=1)
    ent = entropy(probs)
    accept = (pmax >= conf_thr) & (ent <= ent_thr)
    return float(np.mean(accept))


def pick_profile(
    candidates: List[Dict],
    min_acc: float,
    max_ood_accept: float,
    target_coverage: float,
) -> Dict | None:
    feasible = [
        c for c in candidates
        if c["accepted_accuracy"] >= min_acc and c["ood_acceptance"] <= max_ood_accept
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda c: abs(c["coverage"] - target_coverage))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = tf.keras.models.load_model(args.model)

    val_ds = load_split(args.data_root, "validation")
    test_ds = load_split(args.data_root, "test")
    y_val, p_val = collect(model, val_ds)
    y_test, p_test = collect(model, test_ds)

    temp, val_nll = fit_temperature(y_val, p_val)
    p_test_cal = apply_temperature_to_probs(p_test, temp)

    x_ood = load_ood_images(args.ood_dir)
    if x_ood is None:
        x_ood = synthetic_ood(300, args.seed)
    p_ood = model.predict(x_ood, verbose=0)
    p_ood_cal = apply_temperature_to_probs(p_ood, temp)

    grid: List[Dict] = []
    for conf_thr in np.arange(0.60, 0.981, 0.01):
        for ent_thr in np.arange(0.35, 0.951, 0.01):
            stats = accepted_metrics(y_test, p_test_cal, float(conf_thr), float(ent_thr))
            ood_accept = ood_acceptance_rate(p_ood_cal, float(conf_thr), float(ent_thr))
            score = (
                1.4 * stats["accepted_accuracy"]
                + 0.7 * stats["accepted_macro_f1"]
                + 0.5 * stats["coverage"]
                - 5.0 * ood_accept
            )
            grid.append(
                {
                    "confidence_threshold": float(conf_thr),
                    "entropy_threshold": float(ent_thr),
                    **stats,
                    "ood_acceptance": float(ood_accept),
                    "ood_rejection": float(1.0 - ood_accept),
                    "score": float(score),
                }
            )

    grid_sorted = sorted(grid, key=lambda x: x["score"], reverse=True)

    profiles = {
        "conservative": pick_profile(grid_sorted, min_acc=0.995, max_ood_accept=0.05, target_coverage=0.55),
        "balanced": pick_profile(grid_sorted, min_acc=0.985, max_ood_accept=0.10, target_coverage=0.72),
        "permissive": pick_profile(grid_sorted, min_acc=0.970, max_ood_accept=0.18, target_coverage=0.85),
    }

    if profiles["balanced"] is None:
        profiles["balanced"] = grid_sorted[0]

    out = {
        "model": str(args.model),
        "temperature": float(temp),
        "validation_nll_after_temp": float(val_nll),
        "ood_samples": int(len(p_ood_cal)),
        "profiles": profiles,
        "top_candidates": grid_sorted[:40],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 80)
    print("SAFEGUARD TRADEOFF RECALIBRATION COMPLETE")
    print("=" * 80)
    print(json.dumps({"profiles": profiles, "temperature": temp}, indent=2))
    print(f"\nSaved full tradeoff report to: {args.output}")


if __name__ == "__main__":
    main()
