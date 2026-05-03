"""
Generate full per-class confusion matrix + metrics for v5 model.
Run from D:\Code:
  d:\Code\.venv310\Scripts\python scripts/generate_confusion_matrix.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

ALL_CLASSES  = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa", "background"]
DISPLAY_NAMES= ["Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Hispa", "Background"]
IMG_SIZE     = (224, 224)
BATCH        = 32

def file_index(root: Path):
    paths, labels = [], []
    for i, cls in enumerate(ALL_CLASSES):
        d = root / cls
        if not d.exists():
            continue
        for p in d.glob("*.*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                paths.append(str(p))
                labels.append(i)
    return np.array(paths), np.array(labels, np.int32)


def build_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def _load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), y
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def main():
    model_path  = Path("models/rice_disease_model_v5_6class.keras")
    test_dir    = Path("Dataset_clean_6class/test")
    output_json = Path("diagnosis_results/confusion_matrix_v5.json")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    paths, y_true = file_index(test_dir)
    if len(paths) == 0:
        print("ERROR: no test images found.")
        return

    print(f"Test samples: {len(paths)}")
    ds = build_ds(paths, y_true)

    y_pred_probs = model.predict(ds, verbose=1)
    y_pred = y_pred_probs.argmax(axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_CLASSES))))
    cm_list = cm.tolist()

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(ALL_CLASSES))),
        target_names=DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Per-class accuracy
    per_class_accuracy = {}
    for i, name in enumerate(DISPLAY_NAMES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            per_class_accuracy[name] = float((y_pred[class_mask] == i).mean())
        else:
            per_class_accuracy[name] = 0.0

    payload = {
        "model": str(model_path),
        "test_dir": str(test_dir),
        "samples": int(len(y_true)),
        "classes": DISPLAY_NAMES,
        "confusion_matrix": cm_list,
        "overall_accuracy": float((y_pred == y_true).mean()),
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {output_json}")

    # Print summary table
    print("\n=== PER-CLASS METRICS ===")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Samples':>10}")
    print("-" * 60)
    for name in DISPLAY_NAMES:
        r = report.get(name, {})
        print(f"{name:<20} {r.get('precision',0):>10.3f} {r.get('recall',0):>10.3f} "
              f"{r.get('f1-score',0):>10.3f} {int(r.get('support',0)):>10}")
    print(f"\n{'Overall Accuracy':<20} {payload['overall_accuracy']:>10.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print(f"{'':>20}", end="")
    for d in DISPLAY_NAMES:
        print(f"{d[:10]:>12}", end="")
    print()
    for i, row_name in enumerate(DISPLAY_NAMES):
        print(f"{row_name:<20}", end="")
        for val in cm_list[i]:
            print(f"{val:>12}", end="")
        print()


if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
