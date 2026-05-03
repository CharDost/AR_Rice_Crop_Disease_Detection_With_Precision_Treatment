"""
Convert trained Keras model to TensorFlow Lite with INT8 quantization
Target: <40MB model size, <50ms inference on low-end devices
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import sys

# Import the custom layer so Keras can find it if needed when loading the full model
# (kept as fallback in case inference model doesn't exist yet)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_robust_clean import FieldConditionAugmentation  # noqa: F401
except ImportError:
    # Define a no-op pass-through if import fails (e.g., training script not present)
    @tf.keras.utils.register_keras_serializable(package="RiceDisease")
    class FieldConditionAugmentation(tf.keras.layers.Layer):  # type: ignore[no-redef]
        def call(self, inputs, training=None):
            return inputs
        def get_config(self):
            return super().get_config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
# Prefer the inference-only model (no augmentation layers) exported by train_robust_clean.py.
# Falls back to the full model if the inference model doesn't exist yet.
INFERENCE_MODEL_PATH = BASE_DIR / 'models' / 'rice_disease_model_inference.keras'
FULL_MODEL_PATH      = BASE_DIR / 'models' / 'rice_disease_model_robust.keras'
MODEL_PATH = INFERENCE_MODEL_PATH if INFERENCE_MODEL_PATH.exists() else FULL_MODEL_PATH
TFLITE_OUTPUT = BASE_DIR / 'models' / 'rice_disease_model.tflite'
# Use the deduplicated clean dataset for calibration
TRAIN_DIR = BASE_DIR / 'Dataset_clean' / 'train'
IMG_SIZE = (224, 224)
NUM_CALIBRATION_IMAGES = 100


def load_calibration_images():
    """Load representative images for quantization calibration"""
    
    logging.info("Loading calibration images...")
    
    # Get sample images from training set
    image_paths = []
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            class_images = list(class_dir.glob('*.jpg'))[:NUM_CALIBRATION_IMAGES // 4]
            image_paths.extend(class_images)
    
    logging.info(f"Using {len(image_paths)} calibration images")
    
    def representative_dataset():
        for img_path in image_paths:
            # Load image as raw [0, 255] float32 values.
            # IMPORTANT: Do NOT normalize here (no /255.0).
            # The model has a built-in Rescaling(1/255) layer that handles normalization.
            # Passing pre-normalized [0,1] values would cause double normalization and
            # corrupt the INT8 quantization calibration (inputs would be in [0, 0.004] range).
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                target_size=IMG_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Keep raw [0, 255] range — no normalization
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            yield [img_array]
    
    return representative_dataset


def convert_to_tflite_quantized(model_path, output_path):
    """
    Convert Keras model to TFLite with full INT8 quantization
    This provides maximum compression and speed
    """
    
    logging.info("\n" + "=" * 60)
    logging.info("Converting to TensorFlow Lite (INT8 quantization)")
    logging.info("=" * 60)
    
    # Load Keras model
    # We prefer the inference-only model (no augmentation layers) because:
    # 1. It loads cleanly without needing custom_objects.
    # 2. Augmentation layers are training-only no-ops at inference anyway.
    # 3. The TFLite graph is smaller and faster.
    logging.info(f"Loading model from {model_path}")
    if 'inference' in str(model_path):
        # Inference model: no custom layers, loads cleanly
        model = tf.keras.models.load_model(model_path)
    else:
        # Full model with augmentation: must provide custom_objects
        logging.info("Loading full model with FieldConditionAugmentation custom_objects...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"FieldConditionAugmentation": FieldConditionAugmentation},
        )

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Note: We use float32 (DEFAULT optimization) for compatibility.
    # Full INT8 with uint8 I/O requires matching the quantization input type in
    # all inference code (Python + Android). float32 TFLite is safer and still
    # provides good compression from the DEFAULT optimization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Provide representative dataset for quantization calibration
    converter.representative_dataset = load_calibration_images()
    
    # Allow TFLite built-in ops (needed for MobileNetV3/EfficientNet)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Keep float32 I/O for compatibility with existing inference code
    # (Python production_inference.py and Android RiceDiseaseClassifier.kt
    # both pass float32 [0,255] inputs)

    # Convert
    logging.info("Converting model...")
    tflite_model = converter.convert()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Report size
    size_mb = len(tflite_model) / (1024 * 1024)
    logging.info(f"\nTFLite model saved to {output_path}")
    logging.info(f"Model size: {size_mb:.2f} MB")
    
    if size_mb < 40:
        logging.info("✓ Size target met (<40 MB)")
    else:
        logging.warning(f"⚠ Size exceeds target (40 MB), actual: {size_mb:.2f} MB")
    
    return tflite_model


def benchmark_tflite_model(tflite_model_path):
    """Test inference speed of TFLite model"""
    
    logging.info("\n" + "=" * 60)
    logging.info("Benchmarking TFLite model")
    logging.info("=" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logging.info(f"Input shape: {input_details[0]['shape']}")
    logging.info(f"Input type: {input_details[0]['dtype']}")
    logging.info(f"Output shape: {output_details[0]['shape']}")
    logging.info(f"Output type: {output_details[0]['dtype']}")
    
    # Create random input (uint8)
    input_shape = input_details[0]['shape']
    # Use float32 [0,255] range — model input dtype is float32 (has built-in Rescaling)
    input_data = np.random.randint(0, 256, size=input_shape).astype(np.float32)
    
    # Warm-up
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Benchmark
    import time
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
    avg_inference_time = elapsed_time / num_runs
    
    logging.info(f"\nAverage inference time: {avg_inference_time:.2f} ms")
    
    if avg_inference_time < 50:
        logging.info("✓ Speed target met (<50 ms)")
    else:
        logging.warning(f"⚠ Inference slower than target (50 ms), actual: {avg_inference_time:.2f} ms")
    
    return avg_inference_time


def test_accuracy(keras_model_path, tflite_model_path):
    """Compare Keras and TFLite model outputs"""
    
    logging.info("\n" + "=" * 60)
    logging.info("Testing quantization accuracy")
    logging.info("=" * 60)
    
    # Load Keras model
    keras_model = tf.keras.models.load_model(keras_model_path)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load test images
    test_images = []
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))[:5]
            test_images.extend(images)
    
    differences = []

    for img_path in test_images[:20]:  # Test on 20 images
        # Load and preprocess — keep raw [0,255] float32 range;
        # DO NOT normalize here: model has built-in Rescaling(1/255) layer
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)  # [0,255] float32
        
        # Keras prediction (pass raw [0,255] — model rescales internally)
        keras_input = np.expand_dims(img_array, axis=0).astype(np.float32)
        keras_output = keras_model.predict(keras_input, verbose=0)[0]
        
        # TFLite prediction — same float32 [0,255] input
        tflite_input = np.expand_dims(img_array, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Dequantize TFLite output if needed
        if output_details[0]['dtype'] == np.uint8:
            scale, zero_point = output_details[0]['quantization']
            tflite_output = (tflite_output.astype(np.float32) - zero_point) * scale
        
        # Normalize to probabilities
        tflite_output = np.exp(tflite_output) / np.sum(np.exp(tflite_output))
        
        # Calculate difference
        diff = np.mean(np.abs(keras_output - tflite_output))
        differences.append(diff)
    
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    logging.info(f"Average output difference: {avg_diff:.4f}")
    logging.info(f"Maximum output difference: {max_diff:.4f}")
    
    if avg_diff < 0.05:
        logging.info("✓ Quantization quality good (avg diff < 0.05)")
    else:
        logging.warning(f"⚠ Quantization may have reduced accuracy (avg diff: {avg_diff:.4f})")


def main():
    """Main conversion pipeline"""
    
    if not MODEL_PATH.exists():
        logging.error(f"Model not found: {MODEL_PATH}")
        logging.error("Please train the model first using train_model.py")
        return
    
    # Convert to TFLite with INT8 quantization
    tflite_model = convert_to_tflite_quantized(MODEL_PATH, TFLITE_OUTPUT)
    
    # Benchmark inference speed
    inference_time = benchmark_tflite_model(TFLITE_OUTPUT)
    
    # Test quantization accuracy
    test_accuracy(MODEL_PATH, TFLITE_OUTPUT)
    
    logging.info("\n" + "=" * 60)
    logging.info("Conversion complete!")
    logging.info("=" * 60)
    logging.info(f"TFLite model: {TFLITE_OUTPUT}")
    logging.info(f"Ready for Android integration")


if __name__ == '__main__':
    main()
