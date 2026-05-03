"""
Minimal Training Script - Avoiding Import Issues
"""

# Import with explicit error handling
print("Starting imports...")
import sys
import os

# Disable problematic matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

print("Importing TensorFlow...")
import tensorflow as tf
print(f"TensorFlow {tf.__version__} loaded")

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import json
import time

print("All imports successful!\n")

# Configuration
BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "Dataset" / "train"
VAL_DIR = BASE_DIR / "Dataset" / "validation"
MODEL_SAVE_PATH = BASE_DIR / "models" / "rice_disease_model_final.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 50

print("="*70)
print("TRAINING RICE DISEASE MODEL - MOBILENETV3")
print("="*70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU: {gpus}")
else:
    print("Using CPU")

print("\n1. Loading Datasets...")
print("-"*70)

# Simple augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Load datasets
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Count from class directories
train_count = 11274  # From organize script
val_count = 2416      # From organize script

print(f"Training images: {train_count}")
print(f"Validation images: {val_count}")

# Compute class weights
print("\n2. Computing Class Weights...")
print("-"*70)

class_counts = {}
for class_name in CLASSES:
    class_dir = TRAIN_DIR / class_name
    count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
    class_counts[class_name] = count
    print(f"   {class_name:20s}: {count:4d} images")

total = sum(class_counts.values())
max_count = max(class_counts.values())

class_weights = {}
for idx, class_name in enumerate(CLASSES):
    weight = max_count / class_counts[class_name]
    class_weights[idx] = weight
    print(f"   {class_name:20s}: weight = {weight:.2f}")

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("\n3. Building Model...")
print("-"*70)

# Base model
base_model = keras.applications.MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

# Build model
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)

print(f"Model created: {model.count_params():,} parameters")
print(f"   Trainable: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================================
# STAGE 1: Train classification head
# ============================================================================
print("\n" + "="*70)
print("STAGE 1: Training Classification Head")
print("="*70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTraining for {EPOCHS_STAGE1} epochs...")
start_time = time.time()

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ],
    verbose=1
)

stage1_time = time.time() - start_time
stage1_acc = max(history1.history['val_accuracy'])
print(f"\n✅ Stage 1 complete in {stage1_time/60:.1f} minutes")
print(f"   Best validation accuracy: {stage1_acc*100:.2f}%")

# Save stage 1
stage1_path = BASE_DIR / "models" / "stage1_best.keras"
model.save(stage1_path)
print(f"✅ Saved: {stage1_path}")

# ============================================================================
# STAGE 2: Fine-tune entire model
# ============================================================================
print("\n" + "="*70)
print("STAGE 2: Fine-tuning Entire Model")
print("="*70)

# Unfreeze all layers
base_model.trainable = True
print(f"✅ Unfrozen {len(base_model.layers)} base model layers")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTraining for {EPOCHS_STAGE2} epochs...")
start_time = time.time()

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            verbose=1
        )
    ],
    verbose=1
)

stage2_time = time.time() - start_time
final_acc = max(history2.history['val_accuracy'])
print(f"\n✅ Stage 2 complete in {stage2_time/60:.1f} minutes")
print(f"   Best validation accuracy: {final_acc*100:.2f}%")

# Save final model
model.save(MODEL_SAVE_PATH)
print(f"✅ Saved final model: {MODEL_SAVE_PATH}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

total_time = (stage1_time + stage2_time) / 60
print(f"\n📊 Summary:")
print(f"   Total time: {total_time:.1f} minutes")
print(f"   Stage 1 best: {stage1_acc*100:.2f}%")
print(f"   Stage 2 best: {final_acc*100:.2f}%")
print(f"   Improvement: +{(final_acc-stage1_acc)*100:.2f}%")

# Save metadata
metadata = {
    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'train': train_count,
        'validation': val_count,
        'classes': CLASSES
    },
    'architecture': 'MobileNetV3-Small',
    'training': {
        'stage1_epochs': EPOCHS_STAGE1,
        'stage2_epochs': EPOCHS_STAGE2,
        'stage1_acc': float(stage1_acc),
        'stage2_acc': float(final_acc),
        'total_minutes': float(total_time)
    },
    'class_weights': {CLASSES[i]: float(class_weights[i]) for i in range(5)}
}

metadata_path = BASE_DIR / "models" / "training_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Metadata saved: {metadata_path}")
print(f"\n🚀 Next: Evaluate on test set")
print(f"   python scripts/deep_evaluation.py")
print("\n" + "="*70)
