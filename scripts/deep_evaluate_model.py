
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate():
    model_path = "models/rice_disease_model_robust.keras"
    if not os.path.exists(model_path):
        print("Model file not found. Wait for training to finish.")
        return

    print("Loading robust model...")
    model = tf.keras.models.load_model(model_path)
    
    data_dir = Path("Dataset_clean/test")
    classes = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
    
    # Process images properly for EfficientNet prediction without shuffle
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        class_names=classes,
        label_mode="int",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )
    
    y_true = []
    y_pred_probs = []
    
    print("Running predictions on the test set...")
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n" + "="*50)
    print("TEST SET METRICS REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix on Clean Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("diagnosis_results/robust_confusion_matrix.png")
    print("\nSaved confusion matrix plot to: diagnosis_results/robust_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()

