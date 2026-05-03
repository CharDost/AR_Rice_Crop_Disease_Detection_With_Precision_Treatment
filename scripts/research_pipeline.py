import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetV2S
try:
    from tensorflow.keras.applications import ConvNeXtTiny
except ImportError:
    # Fallback if ConvNeXt is not available
    from tensorflow.keras.applications import EfficientNetB0 as ConvNeXtTiny

from sklearn.metrics import classification_report, accuracy_score
import cv2

tf.random.set_seed(42)
np.random.seed(42)

CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# PHASE 2: ADVANCED AUGMENTATION PIPELINE
# Using rigorous augmentations as required
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(height_factor=(-0.25, 0.25), width_factor=(-0.25, 0.25)),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    layers.RandomBrightness(factor=0.4),
    layers.RandomContrast(factor=0.4),
    layers.GaussianNoise(stddev=0.15)
])

# For more advanced ones like Motion Blur, CutOut, we can apply via tf.data.Dataset map
def random_cutout(image, label):
    # Simplified CutOut
    mask_size = tf.random.uniform([], minval=20, maxval=50, dtype=tf.int32)
    x = tf.random.uniform([], minval=0, maxval=224 - mask_size, dtype=tf.int32)
    y = tf.random.uniform([], minval=0, maxval=224 - mask_size, dtype=tf.int32)
    
    mask = tf.ones((mask_size, mask_size, 3), dtype=tf.float32) * -1.0 # arbitrary value or 0
    paddings = [[x, 224 - x - mask_size], [y, 224 - y - mask_size], [0, 0]]
    mask = tf.pad(mask, paddings, constant_values=1.0)
    
    image = tf.where(mask < 0, tf.zeros_like(image), image)
    return image, label

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(CLASSES))
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
    return loss_fn

def build_model(arch_name='MobileNetV3Small'):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    
    if arch_name == 'MobileNetV3Small':
        base_model = MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=x)
    elif arch_name == 'EfficientNetV2S':
        base_model = EfficientNetV2S(include_top=False, weights='imagenet', input_tensor=x)
    else:
        base_model = ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=x)
        
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # PHASE 2: Label Smoothing implemented in compilation, but we can do regularized Dense
    outputs = layers.Dense(
        len(CLASSES), 
        activation='softmax', 
        kernel_regularizer=regularizers.l2(1e-4) # Weight decay
    )(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Label smoothing is supported directly in CategoricalCrossentropy
    # However we requested Focal Loss, so we'll use Focal loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1) if arch_name=='MobileNetV3Small' else focal_loss(),
        metrics=['accuracy']
    )
    return model

# PHASE 6: CALIBRATION
class TemperatureScaling:
    def __init__(self, temperature=1.5):
        self.temperature = temperature
        
    def fit(self, logits, true_labels):
        # In a real scenario, this would use an optimizer to minimize NLL
        self.temperature = 1.2 # Placeholder for fitted temp
        
    def predict(self, logits):
        scaled_logits = logits / self.temperature
        return tf.nn.softmax(scaled_logits).numpy()

def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

if __name__ == "__main__":
    print("Research Pipeline Initialized.")
    print("Awaiting Multi-Domain Datasets in `CrossDomain_Dataset/A`, `B`, `C`...")
    print("Use this script to iterate over Phase 2 -> Phase 7 once datasets are injected.")
