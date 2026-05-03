"""Test TFLite model on REAL dataset images to verify all 6 classes work."""
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL = "android/app/src/main/assets/rice_disease_model.tflite"
DATA  = Path("Dataset_clean_6class/test")

CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa", "background"]

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print("=== Testing TFLite model on real images (0-255 range) ===\n")

for cls_name in CLASSES:
    cls_dir = DATA / cls_name
    if not cls_dir.exists():
        print(f"  {cls_name:25s}  -- MISSING directory")
        continue
    
    imgs = list(cls_dir.glob("*.*"))[:5]  # Test first 5 images per class
    correct = 0
    total = len(imgs)
    
    for img_path in imgs:
        raw = tf.io.read_file(str(img_path))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)  # Raw 0-255
        img = tf.expand_dims(img, 0)
        
        interp.set_tensor(inp_detail['index'], img.numpy())
        interp.invoke()
        probs = interp.get_tensor(out_detail['index'])[0]
        pred = np.argmax(probs)
        true_idx = CLASSES.index(cls_name)
        
        ok = "OK" if pred == true_idx else "WRONG"
        if pred == true_idx:
            correct += 1
    
    acc = correct / max(1, total) * 100
    print(f"  {cls_name:25s}  {correct}/{total} correct  ({acc:.0f}%)")

print("\nDone.")
