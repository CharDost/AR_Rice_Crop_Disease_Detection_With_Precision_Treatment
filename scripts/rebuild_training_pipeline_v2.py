r"""
Rebuild pipeline v2 (without fetching new external datasets).

Implements the requested workflow:
1) Build improved training pipeline with augmentation + cross-validation
2) Retrain final model with optimized hyperparameters
3) Recalibrate safeguards and evaluate coverage/accuracy tradeoffs
4) Validate Android integration with the new model artifact

Usage (from D:\Code):
  python scripts/rebuild_training_pipeline_v2.py

Recommended first run (faster sanity):
  python scripts/rebuild_training_pipeline_v2.py --cv-epochs 4 --final-epochs 20 --cv-folds 3
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASSES)


@dataclass
class TrialConfig:
    learning_rate: float
    dropout: float
    label_smoothing: float
    l2: float
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild training pipeline v2")
    parser.add_argument("--source-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--data-root", type=Path, default=Path("Dataset_clean"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--diagnosis-dir", type=Path, default=Path("diagnosis_results"))
    parser.add_argument("--android-assets", type=Path, default=Path("android/app/src/main/assets"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-epochs", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=35)
    parser.add_argument("--final-finetune-epochs", type=int, default=15)

    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-android-validation", action="store_true")
    parser.add_argument("--no-asset-sync", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def build_file_index(data_root: Path, split_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    filepaths: List[str] = []
    labels: List[int] = []
    for split in split_names:
        split_dir = data_root / split
        for idx, cls in enumerate(CLASSES):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.*"):
                filepaths.append(str(p))
                labels.append(idx)
    return np.array(filepaths), np.array(labels, dtype=np.int32)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_hashes(data_root: Path) -> Dict[str, set[str]]:
    split_names = ["train", "validation", "test"]
    hashes: Dict[str, set[str]] = {}
    for split in split_names:
        current: set[str] = set()
        for cls in CLASSES:
            cls_dir = data_root / split / cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"):
                for path in cls_dir.glob(ext):
                    try:
                        current.add(sha1_file(path))
                    except Exception:
                        continue
        hashes[split] = current
    return hashes


def leakage_report(data_root: Path) -> Dict[str, int]:
    hashes = split_hashes(data_root)
    return {
        "train_validation_exact_hash_overlap": int(len(hashes.get("train", set()) & hashes.get("validation", set()))),
        "train_test_exact_hash_overlap": int(len(hashes.get("train", set()) & hashes.get("test", set()))),
        "validation_test_exact_hash_overlap": int(len(hashes.get("validation", set()) & hashes.get("test", set()))),
    }


def ensure_clean_dataset(data_root: Path, source_root: Path, seed: int) -> Dict[str, int]:
    report = leakage_report(data_root)
    overlap_total = sum(report.values())
    if overlap_total == 0 and data_root.exists():
        print(f"[DATA] {data_root} passes exact-hash leakage check")
        return report

    if not source_root.exists():
        raise FileNotFoundError(
            f"{data_root} fails leakage check and source root {source_root} is missing"
        )

    rebuild_script = Path(__file__).with_name("rebuild_clean_dataset.py")
    if not rebuild_script.exists():
        raise FileNotFoundError(f"Could not find dataset rebuild script: {rebuild_script}")

    print("[DATA] Leakage detected or Dataset_clean missing; rebuilding clean split from source dataset")
    cmd = [
        sys.executable,
        str(rebuild_script),
        "--src",
        str(source_root),
        "--dst",
        str(data_root),
        "--seed",
        str(seed),
        "--hash-mode",
        "exact",
    ]
    subprocess.run(cmd, check=True)

    report = leakage_report(data_root)
    overlap_total = sum(report.values())
    if overlap_total != 0:
        raise RuntimeError(f"Clean dataset rebuild still shows leakage: {report}")

    print(f"[DATA] Rebuilt clean dataset and verified zero exact-hash overlap: {report}")
    return report


def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset_from_arrays(
    paths: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 8192), reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def augmentation_block() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.22),
            tf.keras.layers.RandomZoom(height_factor=0.20, width_factor=0.20),
            tf.keras.layers.RandomTranslation(0.12, 0.12),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.RandomBrightness(0.25),
            tf.keras.layers.GaussianNoise(0.03),
        ],
        name="rebuild_v2_augmentation",
    )


def build_model(cfg: TrialConfig, train_base: bool = False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation_block()(inputs)

    base = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = train_base

    x = base(x, training=False)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(cfg.l2),
    )(x)
    x = tf.keras.layers.Dropout(max(0.15, cfg.dropout - 0.15))(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="rice_rebuild_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")],
    )
    return model


def candidate_trials() -> List[TrialConfig]:
    return [
        TrialConfig(1e-3, 0.45, 0.02, 1e-4, 32),
        TrialConfig(8e-4, 0.50, 0.03, 1.5e-4, 32),
        TrialConfig(6e-4, 0.40, 0.01, 8e-5, 32),
        TrialConfig(5e-4, 0.35, 0.00, 5e-5, 32),
    ]


def run_cross_validation_search(
    paths: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    cv_epochs: int,
    seed: int,
) -> Dict:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    trials = candidate_trials()
    trial_results: List[Dict] = []

    for t_idx, cfg in enumerate(trials, start=1):
        fold_scores: List[float] = []
        print(f"\n[CV] Trial {t_idx}/{len(trials)}: {cfg}")
        for f_idx, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), start=1):
            tf.keras.backend.clear_session()
            model = build_model(cfg, train_base=False)

            ds_train = build_dataset_from_arrays(paths[tr_idx], labels[tr_idx], cfg.batch_size, training=True)
            ds_val = build_dataset_from_arrays(paths[va_idx], labels[va_idx], cfg.batch_size, training=False)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=3,
                    mode="max",
                    restore_best_weights=True,
                    verbose=0,
                )
            ]
            hist = model.fit(ds_train, validation_data=ds_val, epochs=cv_epochs, callbacks=callbacks, verbose=0)
            best_val = float(max(hist.history.get("val_accuracy", [0.0])))
            fold_scores.append(best_val)
            print(f"  Fold {f_idx}/{cv_folds} best val acc: {best_val:.4f}")

        trial_results.append(
            {
                "config": cfg.__dict__,
                "fold_scores": fold_scores,
                "mean_val_accuracy": float(np.mean(fold_scores)),
                "std_val_accuracy": float(np.std(fold_scores)),
            }
        )

    best = max(trial_results, key=lambda x: x["mean_val_accuracy"])
    return {
        "trials": trial_results,
        "best": best,
    }


def class_weights_from_labels(labels: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    total = float(np.sum(counts))
    weights = {i: float(total / (NUM_CLASSES * max(1.0, counts[i]))) for i in range(NUM_CLASSES)}
    return weights


def train_final_model(
    data_root: Path,
    cfg: TrialConfig,
    final_epochs: int,
    finetune_epochs: int,
    seed: int,
    models_dir: Path,
) -> Dict:
    train_paths, train_labels = build_file_index(data_root, ["train"])
    val_paths, val_labels = build_file_index(data_root, ["validation"])
    test_paths, test_labels = build_file_index(data_root, ["test"])

    class_w = class_weights_from_labels(train_labels)

    tf.keras.backend.clear_session()
    model = build_model(cfg, train_base=False)

    ds_train = build_dataset_from_arrays(train_paths, train_labels, cfg.batch_size, training=True)
    ds_val = build_dataset_from_arrays(val_paths, val_labels, cfg.batch_size, training=False)
    ds_test = build_dataset_from_arrays(test_paths, test_labels, cfg.batch_size, training=False)

    stage1_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("\n[TRAIN] Stage 1 (frozen backbone)")
    h1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=final_epochs,
        class_weight=class_w,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    # Fine-tune top MobileNet layers
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "MobileNetV3" in layer.name:
            backbone = layer
            break
    if backbone is not None:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=max(1e-5, cfg.learning_rate * 0.1)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")],
    )

    stage2_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("\n[TRAIN] Stage 2 (partial fine-tuning)")
    h2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=finetune_epochs,
        class_weight=class_w,
        callbacks=stage2_callbacks,
        verbose=1,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    keras_out = models_dir / "rice_disease_model_rebuild_v2.keras"
    model.save(keras_out)

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_blob = converter.convert()
    tflite_out = models_dir / "rice_disease_model_rebuild_v2.tflite"
    with open(tflite_out, "wb") as f:
        f.write(tflite_blob)

    train_eval = model.evaluate(ds_train, verbose=0)
    val_eval = model.evaluate(ds_val, verbose=0)
    test_eval = model.evaluate(ds_test, verbose=0)

    return {
        "keras_model_path": str(keras_out),
        "tflite_model_path": str(tflite_out),
        "class_weights": class_w,
        "stage1_best_val_accuracy": float(max(h1.history.get("val_accuracy", [0.0]))),
        "stage2_best_val_accuracy": float(max(h2.history.get("val_accuracy", [0.0]))),
        "metrics": {
            "train_loss": float(train_eval[0]),
            "train_accuracy": float(train_eval[1]),
            "train_top2": float(train_eval[2]),
            "val_loss": float(val_eval[0]),
            "val_accuracy": float(val_eval[1]),
            "val_top2": float(val_eval[2]),
            "test_loss": float(test_eval[0]),
            "test_accuracy": float(test_eval[1]),
            "test_top2": float(test_eval[2]),
        },
    }


def run_python_script(script_path: Path, args: List[str]) -> int:
    cmd = [sys.executable, str(script_path)] + args
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(script_path.parent.parent))
    return int(proc.returncode)


def sync_model_to_android(tflite_model: Path, android_assets: Path) -> None:
    android_assets.mkdir(parents=True, exist_ok=True)
    dst = android_assets / "rice_disease_model.tflite"
    shutil.copy2(tflite_model, dst)
    print(f"[SYNC] Copied {tflite_model} -> {dst}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    scripts_dir = Path(__file__).parent
    start = time.time()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.diagnosis_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(args.data_root),
        "classes": CLASSES,
        "stages": {},
    }

    best_cfg = candidate_trials()[0]

    if not args.skip_cv:
        print("\n=== STAGE A: Cross-validation + hyperparameter search ===")
        cv_paths, cv_labels = build_file_index(args.data_root, ["train", "validation"])
        cv_result = run_cross_validation_search(
            cv_paths,
            cv_labels,
            cv_folds=args.cv_folds,
            cv_epochs=args.cv_epochs,
            seed=args.seed,
        )
        summary["stages"]["cross_validation"] = cv_result
        best_cfg = TrialConfig(**cv_result["best"]["config"])
    else:
        summary["stages"]["cross_validation"] = {"skipped": True}

    trained = None
    if not args.skip_train:
        print("\n=== STAGE B: Final retraining with best config ===")
        trained = train_final_model(
            data_root=args.data_root,
            cfg=best_cfg,
            final_epochs=args.final_epochs,
            finetune_epochs=args.final_finetune_epochs,
            seed=args.seed,
            models_dir=args.models_dir,
        )
        summary["stages"]["retrain"] = trained

        if not args.no_asset_sync:
            sync_model_to_android(Path(trained["tflite_model_path"]), args.android_assets)
    else:
        summary["stages"]["retrain"] = {"skipped": True, "selected_config": best_cfg.__dict__}

    if not args.skip_calibration:
        print("\n=== STAGE C: Safeguard recalibration + coverage tradeoff evaluation ===")
        model_for_cal = Path(trained["keras_model_path"]) if trained else args.models_dir / "rice_disease_model_rebuild_v2.keras"
        calibration_out = args.models_dir / "calibration_config_rebuild_v2.json"
        rc1 = run_python_script(
            scripts_dir / "calibrate_and_validate.py",
            ["--model", str(model_for_cal), "--data-root", str(args.data_root), "--output", str(calibration_out)],
        )
        safeguard_out = args.diagnosis_dir / "safeguard_eval_rebuild_v2.json"
        rc2 = run_python_script(
            scripts_dir / "evaluate_inference_safeguards.py",
            [
                "--model",
                str(model_for_cal),
                "--calibration",
                str(calibration_out),
                "--test-dir",
                str(args.data_root / "test"),
                "--output",
                str(safeguard_out),
            ],
        )
        summary["stages"]["safeguards"] = {
            "calibration_return_code": rc1,
            "evaluation_return_code": rc2,
            "calibration_config": str(calibration_out),
            "safeguard_report": str(safeguard_out),
        }
    else:
        summary["stages"]["safeguards"] = {"skipped": True}

    if not args.skip_android_validation:
        print("\n=== STAGE D: Android integration validation ===")
        tflite_for_android = (
            Path(trained["tflite_model_path"]) if trained else args.models_dir / "rice_disease_model_rebuild_v2.tflite"
        )
        rc3 = run_python_script(
            scripts_dir / "validate_android_integration.py",
            [
                "--tflite-model",
                str(tflite_for_android),
                "--android-assets",
                str(args.android_assets),
                "--output",
                str(args.diagnosis_dir / "android_integration_validation_rebuild_v2.json"),
            ],
        )
        summary["stages"]["android_validation"] = {"return_code": rc3}
    else:
        summary["stages"]["android_validation"] = {"skipped": True}

    summary["elapsed_minutes"] = (time.time() - start) / 60.0
    summary_out = args.models_dir / "rebuild_v2_pipeline_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("REBUILD V2 PIPELINE COMPLETE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"\nSummary written to: {summary_out}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
