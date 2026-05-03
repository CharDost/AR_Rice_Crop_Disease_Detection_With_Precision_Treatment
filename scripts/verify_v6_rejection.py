"""Verify the NEW V6 rejection logic on all test images."""
import numpy as np
import tensorflow as tf
from pathlib import Path
import math

MODEL = "android/app/src/main/assets/rice_disease_model.tflite"
DATA  = Path("Dataset_clean_6class/test")
CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa", "background"]
BACKGROUND_IDX = 5

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

print("=== V6 Rejection Logic Verification ===\n")
print(f"{'CLASS':20s} {'ACCEPT':>7s} {'REJECT':>7s} {'TOTAL':>6s} {'ACC%':>6s}")
print("-" * 55)

total_accept = 0
total_correct = 0
total_images = 0

for cls_name in CLASSES:
    cls_dir = DATA / cls_name
    if not cls_dir.exists():
        continue
    
    imgs = sorted(cls_dir.glob("*.*"))
    accept = 0
    correct = 0
    true_idx = CLASSES.index(cls_name)
    
    for img_path in imgs:
        raw = tf.io.read_file(str(img_path))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, 0)
        
        interp.set_tensor(inp_d['index'], img.numpy())
        interp.invoke()
        probs = interp.get_tensor(out_d['index'])[0]
        
        top_idx = int(np.argmax(probs))
        max_prob = float(probs[top_idx])
        
        # NEW V6 rejection: only model-BG or garbage
        is_model_bg = (top_idx == BACKGROUND_IDX) and (max_prob >= 0.40)
        is_garbage = max_prob < 0.30
        is_valid = (not is_model_bg) and (not is_garbage)
        
        if cls_name == "background":
            # Background should be REJECTED
            if not is_valid:
                correct += 1
        else:
            # Disease/Healthy should be ACCEPTED and correctly classified
            if is_valid:
                accept += 1
                if top_idx == true_idx:
                    correct += 1
    
    n = len(imgs)
    if cls_name == "background":
        pct = correct / max(1, n) * 100
        print(f"{cls_name:20s} {n-correct:7d} {correct:7d} {n:6d} {pct:5.1f}% rejected")
    else:
        pct = correct / max(1, n) * 100
        print(f"{cls_name:20s} {accept:7d} {n-accept:7d} {n:6d} {pct:5.1f}% correct")
        total_accept += accept
        total_correct += correct
    total_images += n

print("-" * 55)
print(f"Disease accept rate: {total_accept}/{total_images - len(list((DATA/'background').glob('*.*')))} "
      f"({total_accept/(total_images - len(list((DATA/'background').glob('*.*'))))*100:.1f}%)")
print(f"Disease accuracy:    {total_correct}/{total_accept} "
      f"({total_correct/max(1,total_accept)*100:.1f}%)")
