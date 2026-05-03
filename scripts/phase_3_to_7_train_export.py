import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV3Small
from sklearn.metrics import classification_report, confusion_matrix
import math
import json

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = r'D:\Code\Dataset_merged_clustered'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 40
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']

# Phase 3: ENHANCED Robust Augmentation (Strengthened for ≥85% robustness target)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.35),
    layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3)),
    layers.RandomTranslation(height_factor=0.25, width_factor=0.25),
    layers.RandomBrightness(factor=0.45),
    layers.RandomContrast(factor=0.45),
    layers.GaussianNoise(stddev=0.15)
])

def focal_loss_with_label_smoothing(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(CLASSES))
        smooth = label_smoothing / len(CLASSES)
        y_true = y_true * (1.0 - label_smoothing) + smooth
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn

def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    
    base_model = MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Label smoothing can be applied in CategoricalCrossentropy
    outputs = layers.Dense(len(CLASSES), activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss_with_label_smoothing(gamma=2.0, alpha=0.25, label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

def cosine_decay_schedule(epoch, lr):
    decay_epochs = EPOCHS
    alpha = 0.0
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / decay_epochs))
    decayed = (1 - alpha) * cosine_decay + alpha
    return lr * decayed

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def add_corruption(image):
    """Simulate real-world camera degradation: blur, noise, low-light, compression"""
    img_arr = image.numpy()
    if np.random.rand() > 0.4:
        kernel_size = np.random.choice([5, 7, 9])
        img_arr = cv2.GaussianBlur(img_arr, (kernel_size, kernel_size), 0)
    if np.random.rand() > 0.4:
        noise = np.random.normal(0, 35, img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255)
    if np.random.rand() > 0.4:
        low_light = np.random.uniform(0.45, 0.80)
        img_arr = np.clip(img_arr * low_light, 0, 255)
    if np.random.rand() > 0.4:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(15, 50)]
        _, encimg = cv2.imencode('.jpg', img_arr.astype(np.uint8), encode_param)
        img_arr = cv2.imdecode(encimg, 1)
    return img_arr


def get_class_weights(train_root):
    counts = {}
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(train_root, cls)
        if not os.path.exists(cls_dir):
            counts[idx] = 1
            continue
        n = 0
        for name in os.listdir(cls_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                n += 1
        counts[idx] = max(1, n)

    total = sum(counts.values())
    n_classes = len(CLASSES)
    class_weights = {k: total / (n_classes * v) for k, v in counts.items()}
    return class_weights, counts

if __name__ == "__main__":
    train_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'train'), shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'validation'), shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
    test_ds = tf.keras.utils.image_dataset_from_directory(os.path.join(DATA_DIR, 'test'), shuffle=False, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    class_weights, class_counts = get_class_weights(os.path.join(DATA_DIR, 'train'))
    print("Class counts:", class_counts)
    print("Class weights:", class_weights)
    
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.LearningRateScheduler(cosine_decay_schedule, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=r'D:\Code\models\rice_disease_final_mobilenetv3.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    print("\nStarting PHASE 4: Model Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    model.save(r'D:\Code\models\rice_disease_final_mobilenetv3.keras')

    print("\nStarting PHASE 5: Honest Evaluation...")
    clean_loss, clean_acc = model.evaluate(test_ds, verbose=0)
    print(f"Clean Test Accuracy: {clean_acc * 100:.2f}%")

    y_true = []
    y_pred = []
    y_prob = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nRunning Corrupted Robustness Test...")
    y_pred_corrupt = []
    for images, _ in test_ds:
        corrupted_images = np.array([add_corruption(img) for img in images], dtype=np.float32)
        probs = model.predict(corrupted_images, verbose=0)
        y_pred_corrupt.extend(np.argmax(probs, axis=1).tolist())
    y_pred_corrupt = np.array(y_pred_corrupt)
    robust_acc = np.mean(y_pred_corrupt == y_true) * 100
    print(f"Corrupted Test Accuracy: {robust_acc:.2f}%")

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    avg_conf = float(np.mean(np.max(y_prob, axis=1)))
    print(f"ECE: {ece:.4f}")
    print(f"Average Confidence: {avg_conf:.4f}")
    
    print("\nStarting PHASE 7: Export Model (TFLite with Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(r'D:\Code\models\rice_disease_final_quantized.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    print("Exported TF Lite Quantized model offline mobile inference.")

    summary = {
        "clean_accuracy": float(clean_acc),
        "corrupted_accuracy": float(robust_acc / 100.0),
        "ece": float(ece),
        "average_confidence": float(avg_conf),
        "class_counts": class_counts,
        "class_weights": class_weights,
    }
    with open(r'D:\Code\models\final_training_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)

    print("Script setup complete.")
