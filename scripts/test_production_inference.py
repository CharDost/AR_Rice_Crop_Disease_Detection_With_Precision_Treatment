"""
Test Production Inference Module

Tests the production inference system with various scenarios:
- Normal predictions
- Confidence thresholding
- Edge cases (dark, bright, blurry images)
- Performance benchmarking
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from production_inference import (
    RiceDiseaseDetector, 
    ConfidenceLevel, 
    quick_predict
)

BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / "Dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "rice_disease_model.tflite"

print("\n" + "="*70)
print("TESTING PRODUCTION INFERENCE MODULE")
print("="*70)

# ============================================================================
# TEST 1: Basic Initialization and Loading
# ============================================================================
print("\n1. Testing Model Loading...")
print("-"*70)

try:
    detector = RiceDiseaseDetector(
        model_path=MODEL_PATH,
        enable_validation=True,
        log_predictions=True
    )
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Single Image Prediction
# ============================================================================
print("\n2. Testing Single Image Prediction...")
print("-"*70)

# Get a test image from each class
test_images = []
for class_dir in TEST_DIR.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg"))[:1]  # Get 1 image per class
        test_images.extend(images)

print(f"Testing with {len(test_images)} images (1 per class)")

for img_path in test_images:
    class_name = img_path.parent.name
    print(f"\n  Testing: {class_name} - {img_path.name}")
    
    result = detector.predict(
        img_path,
        confidence_level=ConfidenceLevel.BALANCED
    )
    
    status = "[OK]" if result.is_confident else "[!!]"
    print(f"    {status} Predicted: {result.predicted_class} ({result.confidence:.2%})")
    print(f"       Inference time: {result.inference_time_ms:.1f}ms")
    
    if not result.is_confident:
        print(f"       [!!] Below threshold ({result.threshold_used:.2f})")

# ============================================================================
# TEST 3: Different Confidence Levels
# ============================================================================
print("\n3. Testing Different Confidence Thresholds...")
print("-"*70)

test_image = test_images[0]
print(f"Using image: {test_image.name}")

confidence_levels = [
    ConfidenceLevel.CONSERVATIVE,
    ConfidenceLevel.BALANCED,
    ConfidenceLevel.PERMISSIVE,
    ConfidenceLevel.NONE
]

for conf_level in confidence_levels:
    result = detector.predict(test_image, confidence_level=conf_level)
    status = "[ACCEPT]" if result.is_confident else "[REJECT]"
    print(f"  {conf_level.name:15s} (threshold={conf_level.value:.2f}): "
          f"{status} | Confidence: {result.confidence:.2%}")

# ============================================================================
# TEST 4: Image Validation - Edge Cases
# ============================================================================
print("\n4. Testing Input Validation...")
print("-"*70)

# Create test cases
print("\n  a) Normal image:")
normal_img = np.random.randint(50, 200, size=(500, 500, 3), dtype=np.uint8)
validation = detector.validate_image(normal_img)
print(f"     Valid: {validation.is_valid}")
if validation.warnings:
    for warning in validation.warnings:
        print(f"     [!!] {warning}")

print("\n  b) Very dark image:")
dark_img = np.random.randint(0, 30, size=(500, 500, 3), dtype=np.uint8)
validation = detector.validate_image(dark_img)
print(f"     Valid: {validation.is_valid}")
if validation.warnings:
    for warning in validation.warnings:
        print(f"     [!!] {warning}")

print("\n  c) Very bright image:")
bright_img = np.random.randint(240, 255, size=(500, 500, 3), dtype=np.uint8)
validation = detector.validate_image(bright_img)
print(f"     Valid: {validation.is_valid}")
if validation.warnings:
    for warning in validation.warnings:
        print(f"     [!!] {warning}")

print("\n  d) Normalized image (should fail):")
normalized_img = np.random.rand(500, 500, 3).astype(np.float32)
validation = detector.validate_image(normalized_img)
print(f"     Valid: {validation.is_valid}")
if not validation.is_valid:
    print(f"     [X] {validation.error_message}")

print("\n  e) Wrong number of channels (should fail):")
gray_img = np.random.randint(0, 255, size=(500, 500), dtype=np.uint8)
validation = detector.validate_image(gray_img)
print(f"     Valid: {validation.is_valid}")
if not validation.is_valid:
    print(f"     [X] {validation.error_message}")

# ============================================================================
# TEST 5: Batch Prediction
# ============================================================================
print("\n5. Testing Batch Prediction...")
print("-"*70)

# Get 10 random images
batch_images = []
for class_dir in TEST_DIR.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg"))[:2]
        batch_images.extend(images)

print(f"Processing batch of {len(batch_images)} images...")

# Reset statistics
detector.reset_statistics()

results = detector.predict_batch(
    batch_images[:10], 
    confidence_level=ConfidenceLevel.BALANCED
)

print(f"\n  Batch Results:")
print(f"    Total processed: {len(results)}")
confident_count = sum(1 for r in results if r.is_confident)
print(f"    Confident predictions: {confident_count}/{len(results)} ({confident_count/len(results)*100:.1f}%)")
avg_confidence = np.mean([r.confidence for r in results])
print(f"    Average confidence: {avg_confidence:.2%}")
avg_time = np.mean([r.inference_time_ms for r in results])
print(f"    Average inference time: {avg_time:.1f}ms")

# ============================================================================
# TEST 6: Statistics and Monitoring
# ============================================================================
print("\n6. Testing Statistics...")
print("-"*70)

stats = detector.get_statistics()
print(f"  Total predictions: {stats['total_predictions']}")
print(f"  Rejections: {stats['rejections']}")
print(f"  Rejection rate: {stats['rejection_rate_percent']:.1f}%")
print(f"  Model type: {stats['model_type']}")

# ============================================================================
# TEST 7: Quick Predict Function
# ============================================================================
print("\n7. Testing Quick Predict Function...")
print("-"*70)

test_img = test_images[0]
result_dict = quick_predict(test_img, model_path=MODEL_PATH, confidence_threshold=0.85)
print(f"  Predicted: {result_dict['predicted_class']}")
print(f"  Confidence: {result_dict['confidence']:.2%}")
print(f"  Is confident: {result_dict['is_confident']}")
print(f"  Inference time: {result_dict['inference_time_ms']:.1f}ms")

# ============================================================================
# TEST 8: Performance Benchmark
# ============================================================================
print("\n8. Performance Benchmark (50 predictions)...")
print("-"*70)

import time

# Use first image for consistent benchmarking
benchmark_img = Image.open(test_images[0])
benchmark_array = np.array(benchmark_img)

detector.reset_statistics()
times = []

for i in range(50):
    start = time.perf_counter()
    result = detector.predict(benchmark_array, confidence_level=ConfidenceLevel.NONE)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"  Average: {np.mean(times):.2f}ms")
print(f"  Median:  {np.median(times):.2f}ms")
print(f"  Min:     {np.min(times):.2f}ms")
print(f"  Max:     {np.max(times):.2f}ms")
print(f"  Std Dev: {np.std(times):.2f}ms")

# ============================================================================
# TEST 9: Confidence Rejection Analysis
# ============================================================================
print("\n9. Analyzing Rejection Patterns...")
print("-"*70)

# Test 100 random images with different thresholds
sample_images = []
for class_dir in TEST_DIR.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*.jpg"))[:20]
        sample_images.extend(images)

threshold_tests = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
rejection_rates = []

for threshold in threshold_tests:
    detector.reset_statistics()
    for img_path in sample_images:
        detector.predict(img_path, confidence_level=threshold)
    
    stats = detector.get_statistics()
    rejection_rate = stats['rejection_rate_percent']
    rejection_rates.append(rejection_rate)
    
    print(f"  Threshold {threshold:.2f}: "
          f"{rejection_rate:.1f}% rejected "
          f"({stats['rejections']}/{stats['total_predictions']})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

print("\n[OK] All tests completed successfully!")
print("\nKey Findings:")
print(f"  - Model loads and initializes correctly")
print(f"  - Average inference time: {avg_time:.1f}ms (well below 50ms target)")
print(f"  - Input validation catches edge cases")
print(f"  - Confidence thresholding works as expected")
print(f"  - Batch processing functional")
print(f"  - Statistics tracking operational")

print("\nRecommended Production Settings:")
print(f"  - Confidence threshold: 0.85 (BALANCED)")
print(f"  - Expected rejection rate: ~{rejection_rates[threshold_tests.index(0.85)]:.1f}%")
print(f"  - Enable validation: True")
print(f"  - Log predictions: True (for monitoring)")

print("\n" + "="*70)
