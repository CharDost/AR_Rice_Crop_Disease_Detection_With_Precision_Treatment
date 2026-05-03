"""
Production Pipeline v5 - 6-Class EfficientNetB0

Key change: adds a 6th class 'background' trained on real OOD hard-negative
images collected from the field. This forces the model to *learn* what
non-rice images look like instead of relying purely on confidence thresholds.

Classes:
  0 - bacterial_blight
  1 - blast
  2 - brown_spot
  3 - healthy
  4 - hispa
  5 - background   <-- NEW

Usage:
  d:\\Code\\.venv310\\Scripts\\python scripts/production_pipeline_v5_6class.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, log_loss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASE_CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
ALL_CLASSES = DISEASE_CLASSES + ["background"]
NUM_CLASSES = len(ALL_CLASSES)          # 6
IMG_SIZE = (224, 224)
BACKGROUND_IDX = 5

# ---------------------------------------------------------------------------
# Step 1: Build the 6-class dataset on disk
# ---------------------------------------------------------------------------

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_background_split(
    ood_dir: Path,
    dst_root: Path,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, int]:
    """Copy OOD images into Dataset_clean_6class/{train,validation,test}/background/."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images: List[Path] = []
    for p in ood_dir.iterdir():
        if p.suffix.lower() in exts:
            images.append(p)

    # Deduplicate by SHA-1
    seen: Dict[str, Path] = {}
    for p in sorted(images):
        h = sha1_file(p)
        if h not in seen:
            seen[h] = p
    unique = list(seen.values())

    rng = np.random.default_rng(seed)
    idx = np.arange(len(unique))
    rng.shuffle(idx)
    unique = [unique[i] for i in idx]

    n = len(unique)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    n_test  = n - n_train - n_val

    splits = {
        "train":      unique[:n_train],
        "validation": unique[n_train:n_train + n_val],
        "test":       unique[n_train + n_val:],
    }

    counts = {}
    for split, files in splits.items():
        out = dst_root / split / "background"
        out.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            shutil.copy2(src, out / f"background_{i:05d}{src.suffix.lower()}")
        counts[split] = len(files)

    return counts


def build_6class_dataset(
    src_root: Path,
    ood_dir: Path,
    dst_root: Path,
    seed: int,
) -> None:
    """Symlink / copy 5-class splits + add background class."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    # Copy existing 5-class disease data
    for split in ["train", "validation", "test"]:
        for cls in DISEASE_CLASSES:
            src = src_root / split / cls
            dst = dst_root / split / cls
            if src.exists():
                shutil.copytree(src, dst)

    # Add background class from OOD images
    counts = build_background_split(ood_dir, dst_root, seed)
    print(f"[6-CLASS] Background split: {counts}")


# ---------------------------------------------------------------------------
# Step 2: Data loading helpers
# ---------------------------------------------------------------------------

def file_index(root: Path, splits: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    for sp in splits:
        for i, cls in enumerate(ALL_CLASSES):
            d = root / sp / cls
            if not d.exists():
                continue
            for p in d.glob("*.*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    paths.append(str(p))
                    labels.append(i)
    return np.array(paths), np.array(labels, np.int32)


def build_ds(paths: np.ndarray, labels: np.ndarray, batch: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 8192), reshuffle_each_iteration=True)

    def _load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), y

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def class_weights(labels: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    total = float(counts.sum())
    # Upweight background 2x since it is the most important class to get right
    weights = {i: total / (NUM_CLASSES * max(1.0, counts[i])) for i in range(NUM_CLASSES)}
    weights[BACKGROUND_IDX] *= 2.0
    return weights


# ---------------------------------------------------------------------------
# Step 3: Model (EfficientNetB0, 6 outputs)
# ---------------------------------------------------------------------------

def augment_block() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.22),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomTranslation(0.12, 0.12),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.GaussianNoise(0.03),
    ], name="augment_v5")


def build_model(lr: float = 1e-3, dropout: float = 0.45, l2: float = 1e-4, train_base: bool = False) -> tf.keras.Model:
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment_block()(inp)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=IMG_SIZE + (3,),
        weights="imagenet", pooling="avg")
    base.trainable = train_base
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(max(0.15, dropout - 0.15))(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    m = tf.keras.Model(inp, out, name="rice_v5_6class")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")])
    return m


# ---------------------------------------------------------------------------
# Step 4: Calibration
# ---------------------------------------------------------------------------

def apply_temp(probs: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    lg = np.log(p) / max(1e-6, t)
    lg -= lg.max(axis=1, keepdims=True)
    ex = np.exp(lg)
    return ex / ex.sum(axis=1, keepdims=True)


def norm_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalised entropy in [0,1]. Matches calibration_config_v5 thresholds."""
    p = np.clip(probs, 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return h / math.log(NUM_CLASSES)


def fit_temperature(y_val: np.ndarray, p_val: np.ndarray) -> Tuple[float, float]:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.5, 181):
        nll = log_loss(y_val, apply_temp(p_val, float(t)), labels=np.arange(NUM_CLASSES))
        if nll < best_nll:
            best_nll, best_t = float(nll), float(t)
    return best_t, best_nll


def collect_probs(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in ds:
        ps.append(model.predict(x, verbose=0))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def tune_thresholds(y_val: np.ndarray, p_val_cal: np.ndarray) -> Dict[str, Dict]:
    """
    Tune on validation only. For a 6-class model the 'background' class is
    a real class, so we want:
      - High accuracy on disease classes (0-4)
      - High 'background' recall (background images predicted as background)
    We still use a confidence gate as a fallback for any unseen OOD.
    """
    # Separate disease vs background in validation set
    disease_mask = y_val < BACKGROUND_IDX
    bg_mask = y_val == BACKGROUND_IDX

    grid = []
    for conf in np.arange(0.50, 0.96, 0.01):
        for ent in np.arange(0.25, 0.90, 0.01):
            pmax = p_val_cal.max(axis=1)
            ent_v = norm_entropy(p_val_cal)
            pred = p_val_cal.argmax(axis=1)

            # "Accepted" = high confidence OR predicted as background
            accept = (pmax >= conf) & (ent_v <= ent)

            if accept.sum() == 0:
                continue

            # Disease accuracy on accepted disease samples
            d_accept = accept & disease_mask
            if d_accept.sum() > 0:
                d_acc = float(accuracy_score(y_val[d_accept], pred[d_accept]))
            else:
                d_acc = 0.0

            # Coverage on disease samples
            d_cov = float(d_accept.sum() / max(1, disease_mask.sum()))

            # Background recall (how many background samples are correctly predicted as bg)
            if bg_mask.sum() > 0:
                bg_recall = float((pred[bg_mask] == BACKGROUND_IDX).mean())
            else:
                bg_recall = 1.0

            score_con = 1.5 * d_acc + 0.2 * d_cov + 2.0 * bg_recall
            score_bal = 1.2 * d_acc + 0.6 * d_cov + 1.5 * bg_recall
            score_per = 0.9 * d_acc + 1.0 * d_cov + 1.0 * bg_recall

            grid.append(dict(
                conf=float(conf), ent=float(ent),
                d_acc=d_acc, d_cov=d_cov, bg_recall=bg_recall,
                sc=score_con, sb=score_bal, sp=score_per,
            ))

    def pick(key: str, min_acc: float, min_bg_recall: float) -> Dict:
        feasible = [g for g in grid if g["d_acc"] >= min_acc and g["bg_recall"] >= min_bg_recall]
        if not feasible:
            return max(grid, key=lambda g: g[key])
        return max(feasible, key=lambda g: g[key])

    return {
        "conservative": pick("sc", 0.995, 0.90),
        "balanced":     pick("sb", 0.980, 0.85),
        "permissive":   pick("sp", 0.960, 0.75),
    }


def eval_with_rejection(
    y_true: np.ndarray,
    probs: np.ndarray,
    conf: float,
    ent_thr: float,
) -> Dict:
    """Evaluate using confidence gating + background class for OOD rejection."""
    pmax = probs.max(axis=1)
    ent_v = norm_entropy(probs)
    pred = probs.argmax(axis=1)

    disease_mask = y_true < BACKGROUND_IDX
    bg_mask = y_true == BACKGROUND_IDX

    # Threshold-based rejection for disease samples
    accept = (pmax >= conf) & (ent_v <= ent_thr)
    d_accepted = accept & disease_mask

    disease_cov = float(d_accepted.sum() / max(1, disease_mask.sum()))
    d_acc = float(accuracy_score(y_true[d_accepted], pred[d_accepted])) if d_accepted.sum() > 0 else 0.0

    # Background class accuracy
    bg_recall = float((pred[bg_mask] == BACKGROUND_IDX).mean()) if bg_mask.sum() > 0 else 1.0
    bg_precision = 0.0
    if (pred == BACKGROUND_IDX).sum() > 0:
        bg_precision = float((y_true[pred == BACKGROUND_IDX] == BACKGROUND_IDX).mean())

    # Per-class report on disease classes only
    report = classification_report(
        y_true[d_accepted], pred[d_accepted],
        labels=list(range(len(DISEASE_CLASSES))),
        target_names=DISEASE_CLASSES,
        output_dict=True, zero_division=0,
    ) if d_accepted.sum() > 0 else {}

    per_class_recall = {c: float(report[c]["recall"]) for c in DISEASE_CLASSES} if report else {}
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

    return {
        "disease_coverage":    disease_cov,
        "disease_accuracy":    d_acc,
        "disease_macro_f1":    macro_f1,
        "per_class_recall":    per_class_recall,
        "background_recall":   bg_recall,
        "background_precision":bg_precision,
        "conf_thr":            conf,
        "ent_thr":             ent_thr,
    }


# ---------------------------------------------------------------------------
# Step 5: TFLite export
# ---------------------------------------------------------------------------

def export_tflite(keras_path: str, out_path: Path) -> None:
    model = tf.keras.models.load_model(keras_path)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    try:
        blob = conv.convert()
        print(f"[TFLite] float16 export: {len(blob) / 1024:.0f} KB")
    except Exception as e:
        print(f"[TFLite] float16 failed ({e}), falling back to dynamic range")
        conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
        conv2.optimizations = [tf.lite.Optimize.DEFAULT]
        conv2.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        blob = conv2.convert()

    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"[TFLite] saved -> {out_path} ({len(blob) / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",      type=Path, default=Path("Dataset_clean"))
    ap.add_argument("--ood-dir",        type=Path, default=Path("OOD"))
    ap.add_argument("--dst-root",       type=Path, default=Path("Dataset_clean_6class"))
    ap.add_argument("--models-dir",     type=Path, default=Path("models"))
    ap.add_argument("--diag-dir",       type=Path, default=Path("diagnosis_results"))
    ap.add_argument("--android-assets", type=Path, default=Path("android/app/src/main/assets"))
    ap.add_argument("--seed",           type=int,  default=42)
    ap.add_argument("--stage1-epochs",  type=int,  default=35)
    ap.add_argument("--stage2-epochs",  type=int,  default=15)
    ap.add_argument("--batch",          type=int,  default=32)
    ap.add_argument("--lr",             type=float, default=1e-3)
    ap.add_argument("--dropout",        type=float, default=0.45)
    ap.add_argument("--l2",             type=float, default=1e-4)
    ap.add_argument("--skip-dataset",   action="store_true")
    ap.add_argument("--skip-train",     action="store_true")
    ap.add_argument("--skip-android",   action="store_true")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.diag_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    KERAS_OUT  = args.models_dir / "rice_disease_model_v5_6class.keras"
    TFLITE_OUT = args.models_dir / "rice_disease_model_v5_6class.tflite"
    CAL_OUT    = args.models_dir / "calibration_config_v5.json"
    REPORT_OUT = args.diag_dir   / "production_eval_v5.json"

    # --- Step 1: Build 6-class dataset ---
    if not args.skip_dataset:
        print("\n=== Building 6-class dataset ===")
        build_6class_dataset(args.data_root, args.ood_dir, args.dst_root, args.seed)
    else:
        print("[SKIP] dataset build")

    # --- Step 2: Load splits ---
    train_p, train_l = file_index(args.dst_root, ["train"])
    val_p,   val_l   = file_index(args.dst_root, ["validation"])
    test_p,  test_l  = file_index(args.dst_root, ["test"])

    per_class_train = {ALL_CLASSES[i]: int((train_l == i).sum()) for i in range(NUM_CLASSES)}
    print(f"\n[DATA] train={len(train_p)} val={len(val_p)} test={len(test_p)}")
    print(f"[DATA] per-class train: {per_class_train}")

    ds_train = build_ds(train_p, train_l, args.batch, True)
    ds_val   = build_ds(val_p,   val_l,   args.batch, False)
    ds_test  = build_ds(test_p,  test_l,  args.batch, False)

    cw = class_weights(train_l)
    print(f"[DATA] class_weights: { {ALL_CLASSES[i]: round(cw[i], 3) for i in range(NUM_CLASSES)} }")

    # --- Step 3: Train ---
    if not args.skip_train:
        tf.keras.backend.clear_session()
        model = build_model(args.lr, args.dropout, args.l2, train_base=False)

        callbacks_s1 = [
            tf.keras.callbacks.EarlyStopping(
                "val_accuracy", patience=7, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", 0.5, 3, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(
                str(args.models_dir / "v5_stage1_best.keras"),
                save_best_only=True, monitor="val_accuracy"),
        ]

        print("\n=== STAGE 1: frozen backbone ===")
        model.fit(ds_train, validation_data=ds_val, epochs=args.stage1_epochs,
                  class_weight=cw, callbacks=callbacks_s1, verbose=1)

        # Unfreeze top 30 layers of backbone
        for layer in model.layers:
            if hasattr(layer, "layers"):
                layer.trainable = True
                for l in layer.layers[:-30]:
                    l.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(max(1e-5, args.lr * 0.1)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")])

        callbacks_s2 = [
            tf.keras.callbacks.EarlyStopping(
                "val_accuracy", patience=7, restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau("val_loss", 0.5, 3, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(
                str(KERAS_OUT), save_best_only=True, monitor="val_accuracy"),
        ]

        print("\n=== STAGE 2: partial fine-tune ===")
        model.fit(ds_train, validation_data=ds_val, epochs=args.stage2_epochs,
                  class_weight=cw, callbacks=callbacks_s2, verbose=1)
    else:
        print("[SKIP] training - loading existing model")
        model = tf.keras.models.load_model(KERAS_OUT)

    # --- Step 4: Calibrate (val only, test stays BLIND) ---
    print("\n=== Calibrating (val) ===")
    y_val,  p_val  = collect_probs(model, ds_val)
    y_test, p_test = collect_probs(model, ds_test)

    temp, val_nll = fit_temperature(y_val, p_val)
    p_val_cal  = apply_temp(p_val,  temp)
    p_test_cal = apply_temp(p_test, temp)
    print(f"[CAL] temperature={temp:.4f}  val_nll={val_nll:.4f}")

    profiles = tune_thresholds(y_val, p_val_cal)
    for name, prof in profiles.items():
        print(f"[PROFILE:{name}] conf={prof['conf']:.2f} ent={prof['ent']:.2f} "
              f"bg_recall={prof['bg_recall']:.3f}")

    # --- Step 5: Blind test evaluation ---
    print("\n=== Blind test evaluation ===")
    test_results = {}
    for name, prof in profiles.items():
        res = eval_with_rejection(y_test, p_test_cal, prof["conf"], prof["ent"])
        test_results[name] = res
        print(f"\n[TEST:{name}]")
        print(f"  Disease coverage={res['disease_coverage']:.3f}  "
              f"acc={res['disease_accuracy']:.4f}  f1={res['disease_macro_f1']:.4f}")
        print(f"  Background recall={res['background_recall']:.3f}  "
              f"precision={res['background_precision']:.3f}")

    # --- Step 6: Export TFLite ---
    print("\n=== Exporting TFLite ===")
    export_tflite(str(KERAS_OUT), TFLITE_OUT)

    # --- Step 7: Save calibration config ---
    bal = profiles["balanced"]
    cal_payload = {
        "version": "v5",
        "num_classes": NUM_CLASSES,
        "classes": ALL_CLASSES,
        "background_class_index": BACKGROUND_IDX,
        "model": str(KERAS_OUT),
        "temperature": temp,
        "validation_nll_after_temp": val_nll,
        "rejection": {
            "confidence_threshold": bal["conf"],
            "entropy_threshold":    bal["ent"],
            "background_class_index": BACKGROUND_IDX,
            "note": "Predictions with class==background OR (conf<threshold AND ent>threshold) are rejected.",
        },
        "profiles": {
            name: {
                "confidence_threshold": p["conf"],
                "entropy_threshold":    p["ent"],
                "background_recall_val": p["bg_recall"],
            }
            for name, p in profiles.items()
        },
        "android_entropy_note": (
            "Android calculateEntropy() computes RAW Shannon H. "
            f"Balanced python ent_thr={bal['ent']:.3f} -> android raw = "
            f"{bal['ent'] * math.log(NUM_CLASSES):.4f} (ln({NUM_CLASSES})={math.log(NUM_CLASSES):.4f})"
        ),
        "android_raw_entropy_balanced": float(bal["ent"] * math.log(NUM_CLASSES)),
    }
    with open(CAL_OUT, "w", encoding="utf-8") as f:
        json.dump(cal_payload, f, indent=2)
    print(f"[CAL] saved -> {CAL_OUT}")

    # --- Step 8: Full report ---
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":      str(KERAS_OUT),
        "tflite":     str(TFLITE_OUT),
        "calibration": str(CAL_OUT),
        "temperature": temp,
        "classes":    ALL_CLASSES,
        "split_sizes": {
            "train": int(len(train_p)),
            "val":   int(len(val_p)),
            "test":  int(len(test_p)),
        },
        "test_profiles": test_results,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    }
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] saved -> {REPORT_OUT}")

    # --- Step 9: Sync to Android ---
    if not args.skip_android:
        args.android_assets.mkdir(parents=True, exist_ok=True)
        dst = args.android_assets / "rice_disease_model.tflite"
        if dst.exists():
            shutil.copy2(dst, args.android_assets / "rice_disease_model_previous.tflite")
        shutil.copy2(TFLITE_OUT, dst)
        print(f"[ANDROID] model synced -> {dst}")

        # Update labels.txt with 6 classes
        labels_txt = args.android_assets / "labels.txt"
        display_names = [
            "Bacterial Blight", "Blast", "Brown Spot",
            "Healthy", "Hispa", "Not a Rice Leaf",
        ]
        labels_txt.write_text("\n".join(display_names) + "\n", encoding="utf-8")
        print(f"[ANDROID] labels.txt updated with 6 classes -> {labels_txt}")

    print("\n" + "=" * 60)
    print("PRODUCTION PIPELINE V5 (6-CLASS) COMPLETE")
    print("=" * 60)
    bal_res = test_results.get("balanced", {})
    print(f"  Disease coverage:      {bal_res.get('disease_coverage', 0):.1%}")
    print(f"  Disease accuracy:      {bal_res.get('disease_accuracy', 0):.4f}")
    print(f"  Disease macro F1:      {bal_res.get('disease_macro_f1', 0):.4f}")
    print(f"  Background recall:     {bal_res.get('background_recall', 0):.1%}")
    print(f"  Background precision:  {bal_res.get('background_precision', 0):.1%}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
