"""
Simulate the EXACT Android rejection logic on real test images.
This will show us which gate is rejecting Bacterial Blight, Hispa, and Healthy.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import math

MODEL = "android/app/src/main/assets/rice_disease_model.tflite"
DATA  = Path("Dataset_clean_6class/test")
CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa", "background"]

# ── Android thresholds (copied verbatim from RiceDiseaseClassifier.kt) ──
BACKGROUND_CLASS_IDX   = 5
MIN_CONFIDENCE_FOR_VALID = 0.50
MIN_GREEN_RATIO          = 0.08
MIN_CONFIDENCE_MARGIN    = 0.08
MIN_TEXTURE_VARIANCE     = 8.0
MIN_SHARPNESS            = 15.0
MAX_ENTROPY_FOR_VALID    = 0.627   # raw Shannon entropy
CENTER_CROP_RATIO        = 0.78

def green_ratio(img_np):
    """Simulate Android calculateGreenRatio on a 100x100 image."""
    h, w = img_np.shape[:2]
    # Center crop
    cw = int(w * CENTER_CROP_RATIO)
    ch = int(h * CENTER_CROP_RATIO)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = img_np[y0:y0+ch, x0:x0+cw]
    
    # Resize to 100x100 
    crop = tf.image.resize(crop, (100, 100)).numpy().astype(np.uint8)
    
    leaf_pixels = 0
    total = crop.shape[0] * crop.shape[1]
    for y in range(crop.shape[0]):
        for x in range(crop.shape[1]):
            r, g, b = int(crop[y, x, 0]), int(crop[y, x, 1]), int(crop[y, x, 2])
            is_green = g > 60 and r < 180 and b < 150 and (g - r) > 15 and (g - b) > 15
            is_yellow_green = g > 100 and r > 80 and b < 130 and g >= r * 0.7
            is_yellow = r > 120 and g > 100 and b < 100 and r > b and g > b
            is_pale = r > 150 and g > 150 and b > 140 and (g - b) > 5
            is_brown = r > 80 and r > g and r > b and g > 50 and b < 100
            if is_green or is_yellow_green or is_yellow or is_pale or is_brown:
                leaf_pixels += 1
    return leaf_pixels / total

def texture_variance(img_np):
    """Simulate Android calculateTextureVariance."""
    h, w = img_np.shape[:2]
    cw = int(w * CENTER_CROP_RATIO)
    ch = int(h * CENTER_CROP_RATIO)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = img_np[y0:y0+ch, x0:x0+cw]
    crop = tf.image.resize(crop, (50, 50)).numpy()
    
    total_var = 0.0
    count = 0
    for y in range(1, 49):
        for x in range(1, 49):
            curr = float(crop[y, x].mean())
            right = float(crop[y, x+1].mean())
            below = float(crop[y+1, x].mean())
            total_var += abs(curr - right) + abs(curr - below)
            count += 2
    return total_var / count if count > 0 else 0.0

def sharpness(img_np):
    """Simulate Android calculateSharpness."""
    h, w = img_np.shape[:2]
    cw = int(w * CENTER_CROP_RATIO)
    ch = int(h * CENTER_CROP_RATIO)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    crop = img_np[y0:y0+ch, x0:x0+cw]
    crop = tf.image.resize(crop, (64, 64)).numpy()
    
    s = 0.0
    ssq = 0.0
    cnt = 0
    for y in range(1, 63):
        for x in range(1, 63):
            left  = crop[y, x-1].mean()
            right = crop[y, x+1].mean()
            up    = crop[y-1, x].mean()
            down  = crop[y+1, x].mean()
            gx = (right - left) / 3.0
            gy = (down - up) / 3.0
            grad = abs(gx) + abs(gy)
            s += grad
            ssq += grad * grad
            cnt += 1
    if cnt == 0:
        return 0.0
    mean = s / cnt
    return (ssq / cnt) - (mean * mean)

def entropy(probs):
    e = 0.0
    for p in probs:
        if p > 0.0001:
            e -= p * math.log(p)
    return e


# ── Load TFLite ──
interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

print("=" * 90)
print(f"{'CLASS':20s} {'IMG':15s} {'PRED':20s} {'CONF':>6s} {'ENT':>6s} {'MARG':>6s} {'GREEN':>6s} {'TEX':>6s} {'SHARP':>7s} {'VERDICT':>10s}")
print("=" * 90)

reject_reasons = {}

for cls_name in CLASSES:
    cls_dir = DATA / cls_name
    if not cls_dir.exists():
        continue
    
    imgs = sorted(cls_dir.glob("*.*"))[:8]
    
    for img_path in imgs:
        raw = tf.io.read_file(str(img_path))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img_resized = tf.image.resize(img, (224, 224))
        img_float = tf.cast(img_resized, tf.float32)
        img_batch = tf.expand_dims(img_float, 0)
        
        # Get model prediction
        interp.set_tensor(inp_d['index'], img_batch.numpy())
        interp.invoke()
        probs = interp.get_tensor(out_d['index'])[0]
        
        top_idx = np.argmax(probs)
        second_idx = np.argsort(probs)[-2]
        max_prob = float(probs[top_idx])
        second_prob = float(probs[second_idx])
        margin = max_prob - second_prob
        ent = entropy(probs)
        
        # Image quality checks
        img_uint8 = img_resized.numpy().astype(np.uint8)
        gr = green_ratio(img_uint8)
        tv = texture_variance(img_uint8)
        sh = sharpness(img_uint8)
        
        # Android rejection logic
        is_model_bg = (top_idx == BACKGROUND_CLASS_IDX) and (max_prob >= 0.50)
        passes_gate = (
            gr >= MIN_GREEN_RATIO and
            tv >= MIN_TEXTURE_VARIANCE and
            sh >= MIN_SHARPNESS and
            max_prob >= MIN_CONFIDENCE_FOR_VALID and
            ent <= MAX_ENTROPY_FOR_VALID and
            margin >= MIN_CONFIDENCE_MARGIN
        )
        is_valid = (not is_model_bg) and passes_gate
        
        # Which gate failed?
        fails = []
        if is_model_bg:
            fails.append("MODEL_BG")
        if gr < MIN_GREEN_RATIO:
            fails.append(f"GREEN({gr:.3f}<{MIN_GREEN_RATIO})")
        if tv < MIN_TEXTURE_VARIANCE:
            fails.append(f"TEX({tv:.1f}<{MIN_TEXTURE_VARIANCE})")
        if sh < MIN_SHARPNESS:
            fails.append(f"SHARP({sh:.1f}<{MIN_SHARPNESS})")
        if max_prob < MIN_CONFIDENCE_FOR_VALID:
            fails.append(f"CONF({max_prob:.3f}<{MIN_CONFIDENCE_FOR_VALID})")
        if ent > MAX_ENTROPY_FOR_VALID:
            fails.append(f"ENT({ent:.3f}>{MAX_ENTROPY_FOR_VALID})")
        if margin < MIN_CONFIDENCE_MARGIN:
            fails.append(f"MARG({margin:.3f}<{MIN_CONFIDENCE_MARGIN})")
        
        verdict = "ACCEPT" if is_valid else "REJECT"
        
        pred_name = CLASSES[top_idx]
        short_name = img_path.name[:15]
        
        print(f"{cls_name:20s} {short_name:15s} {pred_name:20s} {max_prob:6.3f} {ent:6.3f} {margin:6.3f} {gr:6.3f} {tv:6.1f} {sh:7.1f} {verdict:>10s}")
        
        if fails:
            for f in fails:
                reject_reasons[f.split("(")[0]] = reject_reasons.get(f.split("(")[0], 0) + 1
            print(f"{'':20s} {'':15s} FAILED: {', '.join(fails)}")

print("\n" + "=" * 50)
print("REJECTION REASON COUNTS:")
for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason:15s}: {count}")
