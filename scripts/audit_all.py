
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import imagehash
from PIL import Image

classes = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]
data_dir = Path("Dataset_clean")

def check_leakage():
    print("--- STEP 1: LEAKAGE CHECK ---")
    train_h = set()
    train_hashes_dict = {}
    
    for p in (data_dir / "train").rglob("*.*"):
        if p.is_file():
            try:
                h = str(imagehash.phash(Image.open(p)))
                train_h.add(h)
                train_hashes_dict[h] = p
            except: pass
            
    val_leak, test_leak = 0, 0
    val_tot, test_tot = 0, 0
    
    for p in (data_dir / "validation").rglob("*.*"):
        if p.is_file():
            try:
                img = Image.open(p)
                img.verify()
                img = Image.open(p)
                h = str(imagehash.phash(img))
                val_tot += 1
                if h in train_h: val_leak += 1
            except: pass

    for p in (data_dir / "test").rglob("*.*"):
        if p.is_file():
            try:
                img = Image.open(p)
                img.verify()
                img = Image.open(p)
                h = str(imagehash.phash(img))
                test_tot += 1
                if h in train_h: test_leak += 1
            except: pass
            
    print(f"Validation Perceptual Leaks: {val_leak}/{val_tot} ({(val_leak/max(1,val_tot))*100:.2f}%)")
    print(f"Test Perceptual Leaks: {test_leak}/{test_tot} ({(test_leak/max(1,test_tot))*100:.2f}%)")
    return val_leak, test_leak

def dirty_data_test():
    print("\n--- STEP 3 & 7: DIRTY DATA ROBUSTNESS ---")
    model = tf.keras.models.load_model("models/rice_disease_model_robust.keras")
    
    # Custom generator for dirty data
    def add_noise_blur(image):
        img_arr = image.numpy()
        # Random Gaussian Blur
        if np.random.rand() > 0.5:
            k = np.random.choice([3, 5, 7])
            img_arr = cv2.GaussianBlur(img_arr, (k, k), 0)
        # Random Noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 25, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)
        # Random Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.5, 1.5)
            img_arr = np.clip(alpha * img_arr, 0, 255)
        return img_arr

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "Dataset_clean/test", labels="inferred", class_names=classes,
        label_mode="int", image_size=(224, 224), batch_size=32, shuffle=False
    )
    
    y_true, y_pred_clean, y_pred_dirty = [], [], []
    
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        y_pred_clean.extend(np.argmax(model.predict(images, verbose=0), axis=1))
        
        # Apply dirty mapping
        dirty_images = np.array([add_noise_blur(img) for img in images])
        y_pred_dirty.extend(np.argmax(model.predict(dirty_images, verbose=0), axis=1))
        
    y_true = np.array(y_true)
    y_pred_clean = np.array(y_pred_clean)
    y_pred_dirty = np.array(y_pred_dirty)
    
    clean_acc = np.mean(y_true == y_pred_clean) * 100
    dirty_acc = np.mean(y_true == y_pred_dirty) * 100
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Dirty Accuracy: {dirty_acc:.2f}%")
    print(f"Performance Drop: {clean_acc - dirty_acc:.2f}%")
    
    print("\nDirty Report per class:")
    print(classification_report(y_true, y_pred_dirty, target_names=classes, digits=4))

check_leakage()
dirty_data_test()

