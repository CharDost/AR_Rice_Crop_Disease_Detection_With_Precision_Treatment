"""Quick check: what input range does the deployed TFLite model expect?"""
import numpy as np
import tensorflow as tf

MODEL = "android/app/src/main/assets/rice_disease_model.tflite"

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()

inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print(f"Input shape:  {inp_detail['shape']}")
print(f"Input dtype:  {inp_detail['dtype']}")
print(f"Output shape: {out_detail['shape']}")

# Test with a solid green image (0-255 range)
dummy_255 = np.full((1, 224, 224, 3), 128.0, dtype=np.float32)
interp.set_tensor(inp_detail['index'], dummy_255)
interp.invoke()
out_255 = interp.get_tensor(out_detail['index'])[0]
print(f"\n--- Input range [0-255] (value=128) ---")
print(f"Output probs:  {np.round(out_255, 4)}")
print(f"Max class:     {np.argmax(out_255)} = {out_255.max():.4f}")
print(f"Entropy:       {-np.sum(out_255 * np.log(out_255 + 1e-12)):.4f}")

# Test with same image but [0-1] range
dummy_01 = np.full((1, 224, 224, 3), 128.0/255.0, dtype=np.float32)
interp.set_tensor(inp_detail['index'], dummy_01)
interp.invoke()
out_01 = interp.get_tensor(out_detail['index'])[0]
print(f"\n--- Input range [0-1] (value=0.502) ---")
print(f"Output probs:  {np.round(out_01, 4)}")
print(f"Max class:     {np.argmax(out_01)} = {out_01.max():.4f}")
print(f"Entropy:       {-np.sum(out_01 * np.log(out_01 + 1e-12)):.4f}")

# Compare: which one gives a more confident (lower entropy) prediction?
e255 = -np.sum(out_255 * np.log(out_255 + 1e-12))
e01  = -np.sum(out_01  * np.log(out_01  + 1e-12))
print(f"\n=== CONCLUSION ===")
if e255 < e01:
    print(f"[0-255] gives LOWER entropy ({e255:.4f} vs {e01:.4f}) → model expects 0-255")
else:
    print(f"[0-1] gives LOWER entropy ({e01:.4f} vs {e255:.4f}) → model expects 0-1")
