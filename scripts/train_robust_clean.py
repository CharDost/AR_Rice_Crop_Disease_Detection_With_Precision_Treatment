"""
Train a more robust rice disease classifier on Dataset_clean.

Improvements vs baseline:
- Leakage-resistant dataset split (expects Dataset_clean)
- Stronger augmentation simulating real field conditions:
  * Gaussian noise (low-quality mobile camera sensors)
  * Motion blur (hand tremor during capture)
  * Brightness/contrast shifts (outdoor lighting)
  * Random shadow (leaf occlusion, dappled sunlight)
- Class weighting for imbalance
- Label smoothing
- L2 regularization + increased dropout
- Two-stage transfer learning (frozen → full fine-tune)
- Early stopping + LR scheduling
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
IMG_SIZE = (224, 224)
NUM_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Custom augmentation layer — defined at MODULE LEVEL so Keras can find it
# when loading a saved .keras file (required for TFLite conversion).
# The @register_keras_serializable decorator lets Keras serialize the class
# name and reconstruct it automatically in custom_objects.
# ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="RiceDisease")
class FieldConditionAugmentation(tf.keras.layers.Layer):
    """Applies field-condition augmentations only during training.

    Simulates real farm camera conditions missing from the dataset:
    - Gaussian noise  (cheap phone camera sensor noise)
    - Motion blur     (hand tremor during capture)
    - Random shadow   (dappled sunlight / leaf occlusion)

    During inference (training=False) this layer is a no-op pass-through.
    """

    def call(self, inputs, training=None):
        if not training:
            return inputs

        x = inputs

        # --- Gaussian noise ---
        x = tf.cond(
            tf.random.uniform(()) > 0.5,
            lambda: x + tf.random.normal(tf.shape(x), stddev=0.04 * 255.0),
            lambda: x,
        )

        # --- Motion blur (average of randomly shifted frames) ---
        def apply_blur(img):
            blurred = img
            for _ in range(2):
                dy = tf.random.uniform((), -3, 3, dtype=tf.int32)
                dx = tf.random.uniform((), -3, 3, dtype=tf.int32)
                shifted = tf.roll(img, shift=[dy, dx], axis=[0, 1])
                blurred = (blurred + shifted) / 2.0
            return blurred

        x = tf.cond(
            tf.random.uniform(()) > 0.6,
            lambda: tf.map_fn(apply_blur, x),
            lambda: x,
        )

        # --- Random horizontal shadow strip ---
        def apply_shadow(img):
            h = tf.shape(img)[1]
            w = tf.shape(img)[2]
            top    = tf.random.uniform((), 0, 0.6)
            bot    = top + tf.random.uniform((), 0.1, 0.4)
            top_px = tf.cast(top * tf.cast(h, tf.float32), tf.int32)
            bot_px = tf.minimum(tf.cast(bot * tf.cast(h, tf.float32), tf.int32), h)
            shadow_factor = tf.random.uniform((), 0.4, 0.8)
            before = tf.ones([tf.shape(img)[0], top_px, w, 1])
            shadow = tf.ones([tf.shape(img)[0], bot_px - top_px, w, 1]) * shadow_factor
            after  = tf.ones([tf.shape(img)[0], h - bot_px, w, 1])
            mask   = tf.concat([before, shadow, after], axis=1)
            return img * mask

        x = tf.cond(
            tf.random.uniform(()) > 0.7,
            lambda: apply_shadow(x),
            lambda: x,
        )

        return tf.clip_by_value(x, 0.0, 255.0)

    def get_config(self):
        return super().get_config()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train robust clean-split model")
    parser.add_argument("--data-root", type=Path, default=Path("Dataset_clean"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_datasets(data_root: Path, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        labels="inferred",
        class_names=CLASSES,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "validation",
        labels="inferred",
        class_names=CLASSES,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "test",
        labels="inferred",
        class_names=CLASSES,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    def to_one_hot(x, y):
        return x, tf.one_hot(y, depth=NUM_CLASSES)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(to_one_hot, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(to_one_hot, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(to_one_hot, num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds, test_ds


def get_class_weights(train_dir: Path) -> Dict[int, float]:
    y = []
    for idx, c in enumerate(CLASSES):
        cdir = train_dir / c
        if not cdir.exists():
            continue
        count = sum(1 for _ in cdir.glob("*.*"))
        y.extend([idx] * count)

    y_np = np.array(y, dtype=np.int64)
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_np)
    return {int(i): float(w) for i, w in enumerate(weights)}


def build_model(seed: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    tf.keras.utils.set_random_seed(seed)

    # ------------------------------------------------------------------ #
    # Standard Keras augmentation layers
    # ------------------------------------------------------------------ #
    keras_augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.25),
            tf.keras.layers.RandomContrast(0.30),
            tf.keras.layers.RandomBrightness(0.30),
            tf.keras.layers.RandomTranslation(0.15, 0.15),
        ],
        name="keras_augmentation",
    )

    # FieldConditionAugmentation is defined at MODULE LEVEL above
    # (needed so Keras can find it during deserialization)
    field_augment = FieldConditionAugmentation(name="field_augmentation")

    base = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    # Increased L2 regularization and dropout for better generalization
    l2 = tf.keras.regularizers.l2(2e-4)

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = keras_augment(inputs)
    x = field_augment(x)   # Apply field-condition augmentations
    # EfficientNetB0 expects inputs in [0, 255], no manual rescaling needed
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.4)(x)          # Increased from 0.3
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)          # Increased from 0.2
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2"),
        ],
    )


class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    """Stop early only when BOTH train and val have converged and are close.
    
    Previously this stopped too aggressively after just 8 epochs when validation
    accuracy momentarily hit 98.6%. Now requires the gap to be tight AND
    the targets to be higher, preventing premature stopping before the model
    has learned robust representations.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        gap = abs(acc - val_acc)
        # Only stop if both are high AND closely matched (not overfitting)
        if acc >= 0.990 and val_acc >= 0.980 and gap < 0.015:
            print(f"\nReached convergence target (Train: {acc:.4f}, Val: {val_acc:.4f}, Gap: {gap:.4f})! Stopping.")
            self.model.stop_training = True

def callbacks(out_dir: Path, stage: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / f"robust_{stage}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8 if stage == "stage1" else 15,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        TargetAccuracyCallback(),
        tf.keras.callbacks.CSVLogger(str(out_dir / f"history_{stage}.csv")),
    ]


def evaluate_split(model: tf.keras.Model, ds: tf.data.Dataset, name: str) -> Dict[str, float]:
    loss, acc, top2 = model.evaluate(ds, verbose=0)
    return {
        f"{name}_loss": float(loss),
        f"{name}_accuracy": float(acc),
        f"{name}_top2": float(top2),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = make_datasets(args.data_root, args.batch_size, args.seed)
    class_w = get_class_weights(args.data_root / "train")

    model, base = build_model(args.seed)

    t0 = time.time()

    # Stage 1
    compile_model(model, lr=1e-3)
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=class_w,
        callbacks=callbacks(args.out_dir, "stage1"),
        verbose=1,
    )

    # Stage 2
    base.trainable = True
    compile_model(model, lr=5e-5)
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage2_epochs,
        class_weight=class_w,
        callbacks=callbacks(args.out_dir, "stage2"),
        verbose=1,
    )

    robust_model_path = args.out_dir / "rice_disease_model_robust.keras"
    model.save(robust_model_path)
    print(f"Full training model (with augmentation) saved to {robust_model_path}")

    # ------------------------------------------------------------------ #
    # Export an inference-only model — strips augmentation layers so that
    # TFLite conversion works without needing custom_objects at load time.
    # The EfficientNetB0 + head layers are the same; only the training-only
    # augmentation Sequential and FieldConditionAugmentation are removed.
    # ------------------------------------------------------------------ #
    inference_model_path = args.out_dir / "rice_disease_model_inference.keras"
    # Find the base and head layers (skip augmentation layers at the front)
    aug_layer_names = {"keras_augmentation", "field_augmentation"}
    # Re-build inference model using the trained weights
    # Get layers after augmentation
    inference_input = tf.keras.Input(shape=IMG_SIZE + (3,))
    xi = inference_input
    skipping = True
    for layer in model.layers:
        if layer.name in aug_layer_names or skipping:
            if skipping and layer.name not in aug_layer_names:
                # This is the first non-augmentation layer (the EfficientNetB0 base)
                skipping = False
                xi = layer(xi, training=False)
            # else: still skipping
        else:
            xi = layer(xi)
    inference_model = tf.keras.Model(inference_input, xi, name="rice_disease_inference")
    inference_model.save(inference_model_path)
    print(f"Inference-only model (no augmentation layers) saved to {inference_model_path}")

    metrics = {}
    metrics.update(evaluate_split(model, train_ds, "train"))
    metrics.update(evaluate_split(model, val_ds, "validation"))
    metrics.update(evaluate_split(model, test_ds, "test"))

    summary = {
        "model_path": str(robust_model_path),
        "data_root": str(args.data_root),
        "classes": CLASSES,
        "class_weights": class_w,
        "stage1_best_val_accuracy": float(max(h1.history.get("val_accuracy", [0.0]))),
        "stage2_best_val_accuracy": float(max(h2.history.get("val_accuracy", [0.0]))),
        "elapsed_minutes": float((time.time() - t0) / 60.0),
        "final_metrics": metrics,
    }

    out_json = args.out_dir / "robust_training_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("ROBUST TRAINING COMPLETE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
