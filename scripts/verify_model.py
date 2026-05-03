"""
Verify Model After Preprocessing Fix
Test with correct preprocessing (NO manual normalization)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "rice_disease_model_final.keras"
TEST_DIR = BASE_DIR / "Dataset" / "test"
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']

print("\n" + "="*70)
print("TESTING MODEL WITH CORRECT PREPROCESSING")
print("="*70)

# Load model
print("\n1. Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded")

# Test with random images from each class using CORRECT preprocessing
print("\n2. Testing predictions with CORRECT preprocessing (no manual /255):")
print("-"*70)

all_correct = 0
all_total = 0

for class_idx, class_name in enumerate(CLASSES):
    class_dir = TEST_DIR / class_name
    if not class_dir.exists():
        print(f"⚠️  {class_name}: Directory not found")
        continue
    
    image_files = list(class_dir.glob('*.jpg'))[:10]  # Test 10 images per class
    
    correct = 0
    predictions_summary = {}
    
    for img_path in image_files:
        # Load image - NO NORMALIZATION (model has Rescaling layer)
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = keras.utils.img_to_array(img)  # Values 0-255
        img_array = np.expand_dims(img_array, axis=0)
        # DO NOT divide by 255!
        
        # Predict
        pred = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(pred[0])
        predicted_class = CLASSES[predicted_class_idx]
        confidence = pred[0][predicted_class_idx]
        
        # Track predictions
        predictions_summary[predicted_class] = predictions_summary.get(predicted_class, 0) + 1
        
        if predicted_class == class_name:
            correct += 1
    
    total = len(image_files)
    all_correct += correct
    all_total += total
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Determine result emoji
    if accuracy == 100:
        result = "✅"
    elif accuracy >= 80:
        result = "⚠️"
    else:
        result = "❌"
    
    print(f"\n{result} {class_name.upper()}:")
    print(f"   Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"   Predictions:")
    for pred_cls, count in sorted(predictions_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"      → {pred_cls}: {count}/{total} ({count/total*100:.1f}%)")

# Overall result
print("\n" + "="*70)
overall_accuracy = (all_correct / all_total * 100) if all_total > 0 else 0
print(f"OVERALL ACCURACY: {all_correct}/{all_total} ({overall_accuracy:.1f}%)")

if overall_accuracy >= 95:
    print("✅ MODEL IS WORKING CORRECTLY!")
elif overall_accuracy >= 80:
    print("⚠️  MODEL PERFORMANCE IS ACCEPTABLE BUT NOT PERFECT")
else:
    print("❌ MODEL HAS SIGNIFICANT ISSUES")

print("="*70)

# Now verify with image_dataset_from_directory as well
print("\n3. Cross-checking with image_dataset_from_directory:")
print("-"*70)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

dataset_accuracy = np.mean(y_true == y_pred) * 100
print(f"Accuracy using image_dataset_from_directory: {dataset_accuracy:.2f}%")

if abs(overall_accuracy - dataset_accuracy) < 5:
    print("✅ Both methods give similar results - preprocessing is correct!")
else:
    print("❌ Mismatch between methods - there may still be an issue")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
