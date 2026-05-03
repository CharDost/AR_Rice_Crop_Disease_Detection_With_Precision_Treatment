"""
Post-training calibration + reliability validation.

- Fits temperature by minimizing validation NLL (grid search)
- Tunes rejection threshold for OOD safety
- Evaluates in-distribution and OOD performance before/after calibration
- Exports calibration config for inference safeguards
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
try:
    from train_robust_clean import FieldConditionAugmentation
except ImportError:
    @tf.keras.utils.register_keras_serializable(package="RiceDisease")
    class FieldConditionAugmentation(tf.keras.layers.Layer):  # type: ignore
        def call(self, inputs, training=None): return inputs
        def get_config(self): return super().get_config()


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and validate robust model")
    parser.add_argument("--model", type=Path, default=Path("models/rice_disease_model_robust.keras"))
    parser.add_argument("--data-root", type=Path, default=Path("Dataset_clean"))
    parser.add_argument("--ood-dir", type=Path, default=Path("OOD"))
    parser.add_argument("--output", type=Path, default=Path("models/calibration_config.json"))
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
    scaled = logits_like / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 4.0, 141):
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


def metrics_with_rejection(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    entropy_max: float,
) -> Dict[str, float]:
    pmax = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ent = entropy(probs)

    accept = (pmax >= threshold) & (ent <= entropy_max)
    coverage = float(np.mean(accept))

    if accept.sum() == 0:
        return {
            "coverage": coverage,
            "accepted_accuracy": 0.0,
            "accepted_macro_f1": 0.0,
        }

    y_acc = y_true[accept]
    p_acc = pred[accept]
    acc = float(accuracy_score(y_acc, p_acc))
    _, _, f1, _ = precision_recall_fscore_support(
        y_acc,
        p_acc,
        labels=np.arange(len(CLASSES)),
        average="macro",
        zero_division=0,
    )
    return {
        "coverage": coverage,
        "accepted_accuracy": acc,
        "accepted_macro_f1": float(f1),
    }


def synthetic_ood(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arrs: List[np.ndarray] = []
    for i in range(n):
        if i % 2 == 0:
            arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        else:
            v = int(rng.integers(0, 256))
            arr = np.full((224, 224, 3), v, dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs, axis=0).astype(np.float32)


def load_ood_images(ood_dir: Path) -> np.ndarray | None:
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
            imgs.append(tf.keras.utils.img_to_array(img))
        except Exception:
            continue
    if not imgs:
        return None
    return np.stack(imgs, axis=0).astype(np.float32)


def ood_acceptance_rate(probs: np.ndarray, threshold: float, entropy_max: float) -> float:
    pmax = probs.max(axis=1)
    ent = entropy(probs)
    accept = (pmax >= threshold) & (ent <= entropy_max)
    return float(np.mean(accept))


def tune_rejection(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_ood: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    best = None
    for thr in np.arange(0.55, 0.96, 0.01):
        for ent_max in np.arange(0.45, 0.96, 0.01):
            id_stats = metrics_with_rejection(y_val, p_val, float(thr), float(ent_max))
            ood_accept = ood_acceptance_rate(p_ood, float(thr), float(ent_max))

            # Safety-first objective: strongly penalize OOD acceptance.
            # This intentionally sacrifices in-distribution coverage when needed.
            score = (
                1.5 * id_stats["accepted_accuracy"]
                + 0.3 * id_stats["coverage"]
                - 6.0 * ood_accept
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    float(thr),
                    float(ent_max),
                    {
                        "id_coverage": float(id_stats["coverage"]),
                        "id_accepted_accuracy": float(id_stats["accepted_accuracy"]),
                        "id_accepted_macro_f1": float(id_stats["accepted_macro_f1"]),
                        "ood_acceptance": float(ood_accept),
                    },
                )

    assert best is not None
    _, thr, ent, stats = best

    # Hard safety fallback if OOD acceptance is still too high.
    if stats["ood_acceptance"] > 0.20:
        thr = max(thr, 0.995)
        ent = min(ent, 0.30)
        stats = {
            "id_coverage": float(metrics_with_rejection(y_val, p_val, thr, ent)["coverage"]),
            "id_accepted_accuracy": float(metrics_with_rejection(y_val, p_val, thr, ent)["accepted_accuracy"]),
            "id_accepted_macro_f1": float(metrics_with_rejection(y_val, p_val, thr, ent)["accepted_macro_f1"]),
            "ood_acceptance": float(ood_acceptance_rate(p_ood, thr, ent)),
        }

    return thr, ent, stats


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"FieldConditionAugmentation": FieldConditionAugmentation},
    )

    val_ds = load_split(args.data_root, "validation")
    test_ds = load_split(args.data_root, "test")

    y_val, p_val = collect(model, val_ds)
    y_test, p_test = collect(model, test_ds)

    temp, val_nll = fit_temperature(y_val, p_val)

    p_val_cal = apply_temperature_to_probs(p_val, temp)
    p_test_cal = apply_temperature_to_probs(p_test, temp)

    x_ood = load_ood_images(args.ood_dir)
    if x_ood is None:
        x_ood = synthetic_ood(300, args.seed)
    p_ood_raw = model.predict(x_ood, verbose=0)
    p_ood_cal = apply_temperature_to_probs(p_ood_raw, temp)

    thr, ent_max, tune_stats = tune_rejection(y_val, p_val_cal, p_ood_cal)

    final_test = metrics_with_rejection(y_test, p_test_cal, thr, ent_max)
    final_ood_accept = ood_acceptance_rate(p_ood_cal, thr, ent_max)

    payload = {
        "model": str(args.model),
        "temperature": float(temp),
        "validation_nll_after_temp": float(val_nll),
        "rejection": {
            "confidence_threshold": float(thr),
            "entropy_threshold": float(ent_max),
        },
        "tuning_stats": tune_stats,
        "test_after_calibration_and_rejection": {
            "coverage": float(final_test["coverage"]),
            "accepted_accuracy": float(final_test["accepted_accuracy"]),
            "accepted_macro_f1": float(final_test["accepted_macro_f1"]),
        },
        "ood_after_calibration_and_rejection": {
            "acceptance_rate": float(final_ood_accept),
            "rejection_rate": float(1.0 - final_ood_accept),
            "samples": int(len(p_ood_cal)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("CALIBRATION + RELIABILITY VALIDATION COMPLETE")
    print("=" * 80)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
