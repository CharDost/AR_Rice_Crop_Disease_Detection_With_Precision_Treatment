"""
TFLite Model Verification & Benchmarking
Tests TFLite model accuracy and performance for Android deployment
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
KERAS_MODEL_PATH = BASE_DIR / "models" / "rice_disease_model_final.keras"
TFLITE_MODEL_PATH = BASE_DIR / "models" / "rice_disease_model.tflite"
TEST_DIR = BASE_DIR / "Dataset" / "test"
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']

print("\n" + "="*70)
print("TFLITE MODEL VERIFICATION & BENCHMARKING")
print("="*70)

# Check if TFLite model exists
if not TFLITE_MODEL_PATH.exists():
    print(f"\n❌ TFLite model not found at: {TFLITE_MODEL_PATH}")
    print("Need to convert the model first!")
    exit(1)

print(f"\n✅ TFLite model found: {TFLITE_MODEL_PATH}")
print(f"   Size: {TFLITE_MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")

# Load TFLite model
print("\n1. Loading TFLite model...")
# Disable XNNPACK delegate to avoid runtime errors
interpreter = tf.lite.Interpreter(
    model_path=str(TFLITE_MODEL_PATH),
    experimental_delegates=None,
    num_threads=4
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ TFLite model loaded")
print(f"\n   Input details:")
print(f"      Shape: {input_details[0]['shape']}")
print(f"      Type: {input_details[0]['dtype']}")
print(f"      Quantization: {input_details[0].get('quantization', 'None')}")

print(f"\n   Output details:")
print(f"      Shape: {output_details[0]['shape']}")
print(f"      Type: {output_details[0]['dtype']}")
print(f"      Quantization: {output_details[0].get('quantization', 'None')}")

# Check if it's quantized
is_quantized = input_details[0]['dtype'] != np.float32

if is_quantized:
    print(f"\n⚠️  Model is QUANTIZED (INT8)")
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    print(f"   Input scale: {input_scale}, zero_point: {input_zero_point}")
    print(f"   Output scale: {output_scale}, zero_point: {output_zero_point}")
else:
    print(f"\n✅ Model is FLOAT32 (not quantized)")

# Load Keras model for comparison
print("\n2. Loading Keras model for comparison...")
keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print(f"✅ Keras model loaded")

# Function to preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image - NO normalization (model has Rescaling layer)"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    return img_array

# Function to run TFLite inference
def run_tflite_inference(interpreter, image_array, input_details, output_details):
    """Run inference on TFLite model"""
    # Add batch dimension if needed
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Handle quantized models
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]['quantization']
        # Quantize input
        image_array = image_array / input_scale + input_zero_point
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    else:
        # For float32, just ensure it's float32
        image_array = image_array.astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if needed
    if output_details[0]['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data[0], inference_time

# Test on sample images from each class
print("\n3. Testing TFLite vs Keras predictions...")
print("-"*70)

all_results = {
    'tflite_correct': 0,
    'keras_correct': 0,
    'both_correct': 0,
    'total': 0,
    'inference_times': [],
    'mismatches': []
}

for class_idx, class_name in enumerate(CLASSES):
    class_dir = TEST_DIR / class_name
    if not class_dir.exists():
        continue
    
    # Test 10 images per class
    image_files = list(class_dir.glob('*.jpg'))[:10]
    
    class_results = {
        'tflite_correct': 0,
        'keras_correct': 0,
        'mismatch': 0
    }
    
    for img_path in image_files:
        # Preprocess
        img_array = preprocess_image(img_path)
        
        # TFLite prediction
        tflite_output, inf_time = run_tflite_inference(
            interpreter, img_array, input_details, output_details
        )
        tflite_pred_idx = np.argmax(tflite_output)
        tflite_confidence = tflite_output[tflite_pred_idx]
        
        # Keras prediction
        keras_input = np.expand_dims(img_array, axis=0)
        keras_output = keras_model.predict(keras_input, verbose=0)[0]
        keras_pred_idx = np.argmax(keras_output)
        keras_confidence = keras_output[keras_pred_idx]
        
        # Track results
        all_results['total'] += 1
        all_results['inference_times'].append(inf_time)
        
        if tflite_pred_idx == class_idx:
            class_results['tflite_correct'] += 1
            all_results['tflite_correct'] += 1
        
        if keras_pred_idx == class_idx:
            class_results['keras_correct'] += 1
            all_results['keras_correct'] += 1
        
        if tflite_pred_idx == keras_pred_idx == class_idx:
            all_results['both_correct'] += 1
        
        # Check if predictions match
        if tflite_pred_idx != keras_pred_idx:
            class_results['mismatch'] += 1
            all_results['mismatches'].append({
                'image': img_path.name,
                'true_class': class_name,
                'tflite_pred': CLASSES[tflite_pred_idx],
                'keras_pred': CLASSES[keras_pred_idx],
                'tflite_conf': tflite_confidence,
                'keras_conf': keras_confidence
            })
    
    # Print class results
    tflite_acc = class_results['tflite_correct'] / len(image_files) * 100
    keras_acc = class_results['keras_correct'] / len(image_files) * 100
    
    status = "✅" if class_results['mismatch'] == 0 and tflite_acc == 100 else "⚠️" if tflite_acc >= 80 else "❌"
    
    print(f"\n{status} {class_name.upper()}:")
    print(f"   TFLite:  {class_results['tflite_correct']}/{len(image_files)} ({tflite_acc:.1f}%)")
    print(f"   Keras:   {class_results['keras_correct']}/{len(image_files)} ({keras_acc:.1f}%)")
    if class_results['mismatch'] > 0:
        print(f"   ⚠️  Mismatches: {class_results['mismatch']}")

# Overall statistics
print("\n" + "="*70)
print("4. OVERALL RESULTS")
print("="*70)

tflite_overall_acc = all_results['tflite_correct'] / all_results['total'] * 100
keras_overall_acc = all_results['keras_correct'] / all_results['total'] * 100
agreement_rate = all_results['both_correct'] / all_results['total'] * 100

print(f"\nAccuracy:")
print(f"   TFLite Model: {all_results['tflite_correct']}/{all_results['total']} ({tflite_overall_acc:.2f}%)")
print(f"   Keras Model:  {all_results['keras_correct']}/{all_results['total']} ({keras_overall_acc:.2f}%)")
print(f"   Agreement:    {all_results['both_correct']}/{all_results['total']} ({agreement_rate:.2f}%)")

# Performance metrics
avg_inference_time = np.mean(all_results['inference_times'])
min_inference_time = np.min(all_results['inference_times'])
max_inference_time = np.max(all_results['inference_times'])

print(f"\nInference Speed (TFLite):")
print(f"   Average: {avg_inference_time:.2f} ms")
print(f"   Min:     {min_inference_time:.2f} ms")
print(f"   Max:     {max_inference_time:.2f} ms")

# Speed assessment
if avg_inference_time < 50:
    print(f"   ✅ Speed target met (<50ms)")
elif avg_inference_time < 100:
    print(f"   ⚠️  Acceptable but slower than target ({avg_inference_time:.2f}ms)")
else:
    print(f"   ❌ Too slow for real-time use ({avg_inference_time:.2f}ms)")

# Show mismatches if any
if all_results['mismatches']:
    print(f"\n⚠️  PREDICTION MISMATCHES ({len(all_results['mismatches'])} found):")
    print("-"*70)
    for mismatch in all_results['mismatches'][:5]:  # Show first 5
        print(f"\n   Image: {mismatch['image']}")
        print(f"   True:   {mismatch['true_class']}")
        print(f"   TFLite: {mismatch['tflite_pred']} ({mismatch['tflite_conf']*100:.1f}%)")
        print(f"   Keras:  {mismatch['keras_pred']} ({mismatch['keras_conf']*100:.1f}%)")
    
    if len(all_results['mismatches']) > 5:
        print(f"\n   ... and {len(all_results['mismatches']) - 5} more mismatches")

# Final assessment
print("\n" + "="*70)
print("5. ASSESSMENT & RECOMMENDATIONS")
print("="*70)

if tflite_overall_acc >= 95 and agreement_rate >= 95 and avg_inference_time < 50:
    print("\n✅ TFLITE MODEL IS PRODUCTION READY!")
    print("   - High accuracy")
    print("   - Matches Keras model")
    print("   - Fast inference")
    print("   → Safe to use for Android deployment")
    
elif tflite_overall_acc >= 90 and avg_inference_time < 100:
    print("\n⚠️  TFLITE MODEL IS ACCEPTABLE")
    print("   - Good accuracy but not perfect")
    print("   - May have minor quantization effects")
    print("   → Can use for Android but monitor performance")
    
else:
    print("\n❌ TFLITE MODEL HAS ISSUES")
    
    if tflite_overall_acc < 90:
        print("   ❌ Accuracy too low")
        print("   → Need to reconvert model (check quantization settings)")
    
    if agreement_rate < 90:
        print("   ❌ TFLite doesn't match Keras model")
        print("   → Quantization may be too aggressive")
        print("   → Consider using float16 instead of int8")
    
    if avg_inference_time >= 100:
        print("   ❌ Inference too slow")
        print("   → Check if GPU acceleration is available on target device")
    
    print("\n   RECOMMENDED ACTION: Reconvert TFLite model with adjusted settings")

print("\n" + "="*70)
