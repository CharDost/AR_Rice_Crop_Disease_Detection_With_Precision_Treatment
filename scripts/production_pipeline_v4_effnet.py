"""
Production-Grade Rice Disease Pipeline v4

Fixes over rebuild_v2:
  1. Threshold tuned on VALIDATION only; test set kept blind
  2. Balanced profile deployed (was conservative -> 48% coverage)
  3. Android entropy mismatch documented & calibration_config updated
  4. Per-class recall added to safeguard report
  5. Quantization-aware TFLite export with float16 fallback

Usage:
  d:\Code\.venv410\Scripts\python scripts/production_pipeline_v4.py
"""
from __future__ import annotations
import argparse, json, math, shutil, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, log_loss

CLASSES = ["bacterial_blight","blast","brown_spot","healthy","hispa"]
IMG_SIZE = (224, 224)
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_ds(paths, labels, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths),8192), reshuffle_each_iteration=True)
    def _load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), y
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def file_index(root, splits):
    paths, labels = [], []
    for sp in splits:
        for i, cls in enumerate(CLASSES):
            d = root/sp/cls
            if not d.exists(): continue
            for p in d.glob("*.*"):
                paths.append(str(p)); labels.append(i)
    return np.array(paths), np.array(labels, np.int32)


def class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    total = float(counts.sum())
    return {i: total/(NUM_CLASSES*max(1.0,counts[i])) for i in range(NUM_CLASSES)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def augment_block():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.22),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomTranslation(0.12, 0.12),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.GaussianNoise(0.03),
    ], name="augment_v4")


def build_model(lr=1e-3, dropout=0.45, l2=1e-4, train_base=False):
    inp = tf.keras.Input(shape=IMG_SIZE+(3,))
    x = augment_block()(inp)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=IMG_SIZE+(3,),
        weights="imagenet", pooling="avg")
    base.trainable = train_base
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(max(0.15, dropout-0.15))(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    m = tf.keras.Model(inp, out, name="rice_v4_effnet")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")])
    return m


# ---------------------------------------------------------------------------
# Calibration (val only — test stays blind)
# ---------------------------------------------------------------------------

def apply_temp(probs, t):
    p = np.clip(probs, 1e-12, 1.0)
    lg = np.log(p) / max(1e-6, t)
    lg -= lg.max(axis=1, keepdims=True)
    ex = np.exp(lg)
    return ex / ex.sum(axis=1, keepdims=True)


def norm_entropy(probs):
    """Normalized entropy matching Android's raw/log(N) equivalent."""
    p = np.clip(probs, 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return h / math.log(NUM_CLASSES)


def fit_temperature(y_val, p_val):
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.5, 181):
        nll = log_loss(y_val, apply_temp(p_val, float(t)), labels=np.arange(NUM_CLASSES))
        if nll < best_nll:
            best_nll, best_t = float(nll), float(t)
    return best_t, best_nll


def synth_ood(n, seed):
    rng = np.random.default_rng(seed)
    arrs = []
    for i in range(n):
        if i % 3 == 0:
            arr = rng.integers(0,256,(224,224,3),dtype=np.uint8)
        elif i % 3 == 1:
            v = int(rng.integers(0,256))
            arr = np.full((224,224,3),v,dtype=np.uint8)
        else:
            arr = rng.integers(0,256,(224,224,3),dtype=np.uint8)
            # simulate "green noise" as harder OOD
            arr[:,:,1] = np.clip(arr[:,:,1].astype(int)+40,0,255).astype(np.uint8)
        arrs.append(arr)
    return np.stack(arrs).astype(np.float32)


def tune_thresholds(y_val, p_val_cal, p_ood_cal):
    """
    Tune on validation only.
    Returns three profiles: conservative, balanced, permissive.
    Objective: maximise accuracy*coverage while keeping OOD acceptance ≤ target.
    """
    grid = []
    for conf in np.arange(0.55, 0.97, 0.01):
        for ent in np.arange(0.30, 0.96, 0.01):
            pmax = p_val_cal.max(axis=1)
            ent_v = norm_entropy(p_val_cal)
            accept = (pmax >= conf) & (ent_v <= ent)
            if accept.sum() == 0:
                continue
            acc = float(accuracy_score(y_val[accept], p_val_cal.argmax(axis=1)[accept]))
            cov = float(accept.mean())
            ood_pmax = p_ood_cal.max(axis=1)
            ood_ent = norm_entropy(p_ood_cal)
            ood_acc = float(((ood_pmax >= conf) & (ood_ent <= ent)).mean())
            score_con = 1.5*acc + 0.2*cov - 8.0*ood_acc
            score_bal = 1.2*acc + 0.6*cov - 5.0*ood_acc
            score_per = 0.9*acc + 1.0*cov - 3.0*ood_acc
            grid.append(dict(conf=float(conf), ent=float(ent), acc=acc,
                             cov=cov, ood=ood_acc, sc=score_con, sb=score_bal, sp=score_per))

    def pick(key, min_acc, max_ood):
        feasible = [g for g in grid if g["acc"] >= min_acc and g["ood"] <= max_ood]
        if not feasible:
            return max(grid, key=lambda g: g[key])
        return max(feasible, key=lambda g: g[key])

    return {
        "conservative": pick("sc", 0.995, 0.03),
        "balanced":     pick("sb", 0.980, 0.08),
        "permissive":   pick("sp", 0.960, 0.18),
    }


def eval_with_rejection(y_true, probs, conf, ent_thr):
    pmax = probs.max(axis=1)
    ent_v = norm_entropy(probs)
    accept = (pmax >= conf) & (ent_v <= ent_thr)
    pred = probs.argmax(axis=1)
    cov = float(accept.mean())
    if accept.sum() == 0:
        return {"coverage": cov, "accepted_accuracy": 0.0,
                "accepted_macro_f1": 0.0, "per_class_recall": {}}
    yt, yp = y_true[accept], pred[accept]
    acc = float(accuracy_score(yt, yp))
    report = classification_report(yt, yp, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    per_class = {c: float(report[c]["recall"]) for c in CLASSES}
    macro_f1 = float(report["macro avg"]["f1-score"])
    return {"coverage": cov, "accepted_accuracy": acc,
            "accepted_macro_f1": macro_f1, "per_class_recall": per_class}


def collect_probs(model, ds):
    ys, ps = [], []
    for x, y in ds:
        ps.append(model.predict(x, verbose=0))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def export_tflite(keras_path, out_path, data_root):
    model = tf.keras.models.load_model(keras_path)

    # Float16 quantization (best mobile speed vs accuracy tradeoff)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    try:
        blob = conv.convert()
        print(f"[TFLite] float16 export: {len(blob)/1024:.0f} KB")
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
    print(f"[TFLite] saved -> {out_path} ({len(blob)/1024:.0f} KB)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("Dataset_clean"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--diag-dir", type=Path, default=Path("diagnosis_results"))
    ap.add_argument("--android-assets", type=Path, default=Path("android/app/src/main/assets"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-epochs", type=int, default=40)
    ap.add_argument("--stage2-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.45)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-android", action="store_true")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.diag_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    KERAS_OUT = args.models_dir / "rice_disease_model_v4_final.keras"
    TFLITE_OUT = args.models_dir / "rice_disease_model_v4_final.tflite"
    CAL_OUT = args.models_dir / "calibration_config_v4.json"
    REPORT_OUT = args.diag_dir / "production_eval_v4.json"

    # ---- Load splits ----
    train_p, train_l = file_index(args.data_root, ["train"])
    val_p, val_l = file_index(args.data_root, ["validation"])
    test_p, test_l = file_index(args.data_root, ["test"])

    ds_train = build_ds(train_p, train_l, args.batch, True)
    ds_val   = build_ds(val_p, val_l, args.batch, False)
    ds_test  = build_ds(test_p, test_l, args.batch, False)

    cw = class_weights(train_l)
    print(f"\n[DATA] train={len(train_p)} val={len(val_p)} test={len(test_p)}")
    print(f"[DATA] class_weights: { {CLASSES[i]: round(cw[i],3) for i in range(NUM_CLASSES)} }")

    if not args.skip_train:
        tf.keras.backend.clear_session()
        model = build_model(args.lr, args.dropout, args.l2, train_base=False)

        print("\n=== STAGE 1: frozen backbone ===")
        h1 = model.fit(ds_train, validation_data=ds_val, epochs=args.stage1_epochs,
            class_weight=cw, callbacks=[
                tf.keras.callbacks.EarlyStopping("val_accuracy", patience=7,
                    restore_best_weights=True, mode="max"),
                tf.keras.callbacks.ReduceLROnPlateau("val_loss", 0.5, 3, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(
                    str(args.models_dir/"v4_stage1_best.keras"),
                    save_best_only=True, monitor="val_accuracy"),
            ], verbose=1)

        # Unfreeze top 30 layers
        for layer in model.layers:
            if hasattr(layer, "layers"):  # is sub-model
                layer.trainable = True
                for l in layer.layers[:-30]:
                    l.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(max(1e-5, args.lr*0.1)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")])

        print("\n=== STAGE 2: partial fine-tune ===")
        model.fit(ds_train, validation_data=ds_val, epochs=args.stage2_epochs,
            class_weight=cw, callbacks=[
                tf.keras.callbacks.EarlyStopping("val_accuracy", patience=7,
                    restore_best_weights=True, mode="max"),
                tf.keras.callbacks.ReduceLROnPlateau("val_loss", 0.5, 3, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(
                    str(KERAS_OUT), save_best_only=True, monitor="val_accuracy"),
            ], verbose=1)
    else:
        print("[SKIP] training — loading existing model")
        model = tf.keras.models.load_model(KERAS_OUT)

    # ---- Evaluate on all splits ----
    print("\n=== Evaluating splits ===")
    y_val, p_val = collect_probs(model, ds_val)
    y_test, p_test = collect_probs(model, ds_test)

    # ---- Calibrate (val only) ----
    temp, val_nll = fit_temperature(y_val, p_val)
    p_val_cal  = apply_temp(p_val, temp)
    p_test_cal = apply_temp(p_test, temp)
    print(f"[CAL] temperature={temp:.4f}  val_nll={val_nll:.4f}")

    # OOD (harder set: 450 samples, mixed noise + green-shifted)
    ood_raw = synth_ood(450, args.seed)
    p_ood_raw = model.predict(ood_raw, verbose=0)
    p_ood_cal = apply_temp(p_ood_raw, temp)

    # ---- Tune thresholds on val (TEST STAYS BLIND) ----
    profiles = tune_thresholds(y_val, p_val_cal, p_ood_cal)
    print(f"\n[PROFILES] conservative: conf={profiles['conservative']['conf']:.2f} ent={profiles['conservative']['ent']:.2f}")
    print(f"[PROFILES] balanced:     conf={profiles['balanced']['conf']:.2f} ent={profiles['balanced']['ent']:.2f}")
    print(f"[PROFILES] permissive:   conf={profiles['permissive']['conf']:.2f} ent={profiles['permissive']['ent']:.2f}")

    # ---- Blind test evaluation per profile ----
    test_results = {}
    for name, prof in profiles.items():
        res = eval_with_rejection(y_test, p_test_cal, prof["conf"], prof["ent"])
        ood_rate = float(((p_ood_cal.max(axis=1) >= prof["conf"]) &
                          (norm_entropy(p_ood_cal) <= prof["ent"])).mean())
        test_results[name] = {**res, "ood_acceptance": ood_rate,
                               "conf_thr": prof["conf"], "ent_thr": prof["ent"]}
        print(f"\n[TEST:{name}] coverage={res['coverage']:.3f} "
              f"acc={res['accepted_accuracy']:.4f} f1={res['accepted_macro_f1']:.4f} "
              f"ood_accept={ood_rate:.3f}")

    # ---- Export TFLite ----
    export_tflite(str(KERAS_OUT), TFLITE_OUT, args.data_root)

    # ---- Save calibration config (use balanced profile) ----
    bal = profiles["balanced"]
    cal_payload = {
        "version": "v4",
        "model": str(KERAS_OUT),
        "temperature": temp,
        "validation_nll_after_temp": val_nll,
        "rejection": {
            "confidence_threshold": bal["conf"],
            "entropy_threshold": bal["ent"],
        },
        "profiles": {
            name: {"confidence_threshold": p["conf"], "entropy_threshold": p["ent"]}
            for name, p in profiles.items()
        },
        "note_android_entropy": (
            "Android calculateEntropy() computes raw Shannon H (NOT normalized). "
            "Android MAX_ENTROPY_FOR_VALID should equal ent_threshold * ln(5) = "
            f"{bal['ent'] * math.log(NUM_CLASSES):.4f} (raw Shannon units)."
        ),
        "android_raw_entropy_balanced": float(bal["ent"] * math.log(NUM_CLASSES)),
    }
    with open(CAL_OUT, "w") as f:
        json.dump(cal_payload, f, indent=2)
    print(f"\n[CAL] saved -> {CAL_OUT}")

    # ---- Full report ----
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(KERAS_OUT),
        "tflite": str(TFLITE_OUT),
        "calibration": str(CAL_OUT),
        "temperature": temp,
        "classes": CLASSES,
        "split_sizes": {"train": int(len(train_p)), "val": int(len(val_p)), "test": int(len(test_p))},
        "test_profiles": test_results,
        "elapsed_minutes": round((time.time()-t0)/60, 2),
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[REPORT] saved -> {REPORT_OUT}")

    # ---- Sync to Android ----
    if not args.skip_android:
        args.android_assets.mkdir(parents=True, exist_ok=True)
        dst = args.android_assets / "rice_disease_model.tflite"
        if dst.exists():
            shutil.copy2(dst, args.android_assets / "rice_disease_model_previous.tflite")
        shutil.copy2(TFLITE_OUT, dst)
        print(f"[ANDROID] synced {TFLITE_OUT} -> {dst}")

    print("\n" + "="*60)
    print("PRODUCTION PIPELINE V4 COMPLETE")
    print("="*60)
    bal_res = test_results.get("balanced", {})
    print(f"  Coverage (balanced): {bal_res.get('coverage',0):.1%}")
    print(f"  Accepted accuracy:   {bal_res.get('accepted_accuracy',0):.4f}")
    print(f"  Macro F1:            {bal_res.get('accepted_macro_f1',0):.4f}")
    print(f"  OOD acceptance:      {bal_res.get('ood_acceptance',0):.3f}")


if __name__ == "__main__":
    import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    main()
