"""
Deep Model Analysis - Finding the Problem
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "rice_disease_model_final.keras"
TEST_DIR = BASE_DIR / "Dataset" / "test"
CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']

print("\n" + "="*70)
print("DETAILED MODEL ANALYSIS - Finding the Issue")
print("="*70)

# Load model
print("\n1. Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded")

# Check model architecture
print("\n2. Model Architecture:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Total parameters: {model.count_params():,}")

# Test with a few random images from each class
print("\n3. Testing predictions on random images from each class:")
print("-"*70)

for class_idx, class_name in enumerate(CLASSES):
    class_dir = TEST_DIR / class_name
    if not class_dir.exists():
        print(f"⚠️  {class_name}: Directory not found")
        continue
    
    image_files = list(class_dir.glob('*.jpg'))[:5]  # Test 5 images per class
    
    predictions_for_class = []
    
    for img_path in image_files:
        # Load and preprocess
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Predict
        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred[0])
        confidence = pred[0][predicted_class]
        
        predictions_for_class.append({
            'true_class': class_name,
            'predicted_class': CLASSES[predicted_class],
            'confidence': confidence,
            'all_probs': pred[0]
        })
    
    # Analyze results for this class
    correct = sum(1 for p in predictions_for_class if p['predicted_class'] == class_name)
    avg_confidence = np.mean([p['confidence'] for p in predictions_for_class])
    
    print(f"\n{class_name.upper()}:")
    print(f"   Tested: {len(predictions_for_class)} images")
    print(f"   Correct: {correct}/{len(predictions_for_class)} ({correct/len(predictions_for_class)*100:.1f}%)")
    print(f"   Avg confidence: {avg_confidence*100:.1f}%")
    
    # Show what it's predicting
    prediction_counts = {}
    for p in predictions_for_class:
        pred_cls = p['predicted_class']
        prediction_counts[pred_cls] = prediction_counts.get(pred_cls, 0) + 1
    
    print(f"   Predictions breakdown:")
    for pred_cls, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      → {pred_cls}: {count}/{len(predictions_for_class)}")
    
    # Show average probabilities for all classes
    avg_probs = np.mean([p['all_probs'] for p in predictions_for_class], axis=0)
    print(f"   Average probabilities across all classes:")
    for idx, cls in enumerate(CLASSES):
        print(f"      {cls}: {avg_probs[idx]*100:.1f}%")

# Full test set evaluation
print("\n" + "="*70)
print("4. FULL TEST SET EVALUATION")
print("="*70)

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

print("\nPer-class accuracy:")
for i, class_name in enumerate(CLASSES):
    mask = y_true == i
    if mask.sum() > 0:
        class_acc = np.mean(y_pred[mask] == i)
        total = mask.sum()
        correct = (y_pred[mask] == i).sum()
        
        # Show what it's actually predicting for this class
        predictions_for_class = y_pred[mask]
        unique, counts = np.unique(predictions_for_class, return_counts=True)
        
        print(f"\n{class_name}:")
        print(f"   Accuracy: {class_acc*100:.1f}% ({correct}/{total})")
        print(f"   What model predicts for these images:")
        for pred_idx, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            print(f"      → {CLASSES[pred_idx]}: {count}/{total} ({count/total*100:.1f}%)")

# Check if model is always predicting the same class
print("\n" + "="*70)
print("5. PREDICTION DISTRIBUTION")
print("="*70)

unique, counts = np.unique(y_pred, return_counts=True)
print("\nWhat the model predicts across ALL test images:")
for pred_idx, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
    print(f"   {CLASSES[pred_idx]}: {count}/{len(y_pred)} ({count/len(y_pred)*100:.1f}%)")

# Overall accuracy
overall_acc = np.mean(y_true == y_pred)
print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
