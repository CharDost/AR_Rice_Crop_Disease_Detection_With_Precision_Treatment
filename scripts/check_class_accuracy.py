
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from pathlib import Path

def main():
    print("Loading test data and model to calculate exact per-class accuracy...")
    model = tf.keras.models.load_model("models/rice_disease_model_robust.keras")
    classes = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset_clean/test",
        labels="inferred",
        class_names=classes,
        label_mode="int",
        image_size=(224, 224),
        batch_size=64,
        shuffle=False
    )

    y_true = []
    y_pred_probs = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)

    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*50)
    print("EXACT ACCURACY PER CLASS")
    print("="*50)
    
    for i, cls in enumerate(classes):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = (correct / total) * 100
        print(f"{cls:>18} : {accuracy:>6.2f}%  ({correct}/{total} correctly predicted)")
        
    print("="*50)

if __name__ == "__main__":
    main()

