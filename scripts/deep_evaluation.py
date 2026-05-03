"""
Comprehensive evaluation to understand why accuracy is stuck at 67%
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "rice_disease_model_final.keras"
TEST_DIR = BASE_DIR / "Dataset" / "test"
OUTPUT_DIR = BASE_DIR / "diagnosis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']
IMG_SIZE = (224, 224)

print("\n" + "="*70)
print("DEEP EVALUATION - Understanding the 67% Problem")
print("="*70)

# Load model
print("\n1. Loading model...")
model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from: {MODEL_PATH}")

# Load test data
print("\n2. Loading test dataset...")
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

print(f"✅ Test batches: {len(test_ds)}")

# Get predictions
print("\n3. Running predictions on test set...")
y_true = []
y_pred = []
y_pred_probs = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred_probs.extend(predictions)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_probs = np.array(y_pred_probs)

print(f"✅ Predictions complete: {len(y_true)} samples")

# Overall metrics
print("\n" + "="*70)
print("4. Overall Performance")
print("="*70)

accuracy = np.mean(y_true == y_pred)
print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%\n")

# Per-class accuracy
print("Per-Class Accuracy:")
print("-"*70)
for i, class_name in enumerate(CLASSES):
    mask = y_true == i
    if mask.sum() > 0:
        class_acc = np.mean(y_pred[mask] == i)
        total = mask.sum()
        correct = (y_pred[mask] == i).sum()
        print(f"{class_name:20s}: {class_acc*100:5.1f}% ({correct:3d}/{total:3d} correct)")

# Confusion Matrix
print("\n5. Confusion Matrix Analysis")
print("-"*70)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: confusion_matrix.png")

# Normalized confusion matrix
plt.figure(figsize=(12, 10))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES, vmin=0, vmax=1)
plt.title('Normalized Confusion Matrix (% of True Class)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: confusion_matrix_normalized.png")

# Find biggest confusion pairs
print("\n6. Top Confusion Pairs (Where Model Gets Confused)")
print("-"*70)

confusion_pairs = []
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append({
                'true': CLASSES[i],
                'predicted': CLASSES[j],
                'count': cm[i, j],
                'percentage': cm[i, j] / cm[i].sum() * 100
            })

confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

print("\nTop 10 misclassification patterns:")
for idx, pair in enumerate(confusion_pairs[:10], 1):
    print(f"{idx:2d}. {pair['true']:20s} → {pair['predicted']:20s}: "
          f"{pair['count']:3d} samples ({pair['percentage']:5.1f}% of true class)")

# Analyze prediction confidence
print("\n7. Prediction Confidence Analysis")
print("-"*70)

correct_mask = y_true == y_pred
correct_confidences = np.max(y_pred_probs[correct_mask], axis=1)
incorrect_confidences = np.max(y_pred_probs[~correct_mask], axis=1)

print(f"\nCorrect predictions:")
print(f"   Mean confidence: {correct_confidences.mean():.3f}")
print(f"   Median confidence: {np.median(correct_confidences):.3f}")
print(f"   Min confidence: {correct_confidences.min():.3f}")

print(f"\nIncorrect predictions:")
print(f"   Mean confidence: {incorrect_confidences.mean():.3f}")
print(f"   Median confidence: {np.median(incorrect_confidences):.3f}")
print(f"   Max confidence: {incorrect_confidences.max():.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(correct_confidences, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('Confidence: Correct Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.axvline(correct_confidences.mean(), color='red', linestyle='--', label=f'Mean: {correct_confidences.mean():.3f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(incorrect_confidences, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.title('Confidence: Incorrect Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.axvline(incorrect_confidences.mean(), color='blue', linestyle='--', label=f'Mean: {incorrect_confidences.mean():.3f}')
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_distribution.png", dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: confidence_distribution.png")

# Find worst performing samples
print("\n8. Identifying Problematic Images")
print("-"*70)

# Get file paths
test_files = []
for class_idx, class_name in enumerate(CLASSES):
    class_dir = TEST_DIR / class_name
    files = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
    test_files.extend([(f, class_idx) for f in files])

# Find high-confidence errors
high_conf_errors = []
for idx in range(len(y_true)):
    if y_true[idx] != y_pred[idx]:
        conf = y_pred_probs[idx][y_pred[idx]]
        if conf > 0.7:  # High confidence but wrong
            high_conf_errors.append({
                'index': idx,
                'true': CLASSES[y_true[idx]],
                'pred': CLASSES[y_pred[idx]],
                'confidence': conf,
                'file': test_files[idx][0] if idx < len(test_files) else None
            })

high_conf_errors.sort(key=lambda x: x['confidence'], reverse=True)

print(f"\nFound {len(high_conf_errors)} high-confidence errors (conf > 0.7):")
print("\nTop 20 worst mistakes (model very confident but wrong):")
for idx, error in enumerate(high_conf_errors[:20], 1):
    print(f"{idx:2d}. True: {error['true']:20s} → Pred: {error['pred']:20s} "
          f"(conf: {error['confidence']:.3f})")

# Visualize worst mistakes
print("\n9. Visualizing Worst Mistakes")
print("-"*70)

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for idx, error in enumerate(high_conf_errors[:20]):
    if error['file']:
        img = cv2.imread(str(error['file']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        title = f"True: {error['true']}\nPred: {error['pred']}\nConf: {error['confidence']:.2f}"
        axes[idx].set_title(title, fontsize=9, color='red' if error['confidence'] > 0.9 else 'orange')

plt.suptitle('Top 20 High-Confidence Errors', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "worst_mistakes.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: worst_mistakes.png")

# Classification report
print("\n10. Detailed Classification Report")
print("-"*70)
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

# Save report to file
report_path = OUTPUT_DIR / "evaluation_report.txt"
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
    f.write("Per-Class Accuracy:\n")
    f.write("-"*70 + "\n")
    for i, class_name in enumerate(CLASSES):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = np.mean(y_pred[mask] == i)
            total = mask.sum()
            correct = (y_pred[mask] == i).sum()
            f.write(f"{class_name:20s}: {class_acc*100:5.1f}% ({correct:3d}/{total:3d})\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("Top Confusion Pairs:\n")
    f.write("-"*70 + "\n")
    for idx, pair in enumerate(confusion_pairs[:10], 1):
        f.write(f"{idx:2d}. {pair['true']:20s} → {pair['predicted']:20s}: "
                f"{pair['count']:3d} ({pair['percentage']:5.1f}%)\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("Classification Report:\n")
    f.write("-"*70 + "\n")
    f.write(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

print(f"\n✅ Full report saved: {report_path}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print(f"\n📁 All results saved in: {OUTPUT_DIR}")
print("\n🔍 Key files:")
print(f"   - confusion_matrix.png - Shows where model confuses classes")
print(f"   - worst_mistakes.png - Visual inspection of problematic images")
print(f"   - confidence_distribution.png - Model confidence analysis")
print(f"   - evaluation_report.txt - Complete text report")

print("\n💡 Next Steps:")
print("   1. Check confusion_matrix.png to see which classes are confused")
print("   2. Review worst_mistakes.png for data quality issues")
print("   3. Based on findings, decide on improvements:")
print("      - If data quality issues: Manual cleaning needed")
print("      - If classes truly similar: Try better architecture (EfficientNet)")
print("      - If low confidence: More training data or augmentation")
