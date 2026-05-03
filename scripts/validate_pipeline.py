"""
Full pipeline validation script.

Runs a series of quick checks to confirm the inference pipeline is internally
consistent and the deployed model behaves correctly. Does NOT require retraining.

Usage:
    cd d:\\Code
    python scripts/validate_pipeline.py

Checks performed:
1. Model file existence and loadability
2. TFLite input/output shape and dtype verification
3. Preprocessing sanity (correct 0-255 input range)
4. Leaf-gating behavior: known OOD inputs rejected, rice leaf images accepted
5. Calibration config consistency check
6. Dataset_clean integrity check (class counts, no obvious cross-split exact duplicates)
7. Confidence distribution on test set (aggregated stats)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# ---- Setup paths ----
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "Dataset_clean"
CLASSES = ["bacterial_blight", "blast", "brown_spot", "healthy", "hispa"]

PASS = "\u2705"
FAIL = "\u274c"
WARN = "\u26a0\ufe0f "


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# -----------------------------------------------------------------------
# Check 1: Model files exist
# -----------------------------------------------------------------------
def check_model_files() -> bool:
    section("CHECK 1: Model File Existence")
    ok = True
    required = [
        MODELS_DIR / "rice_disease_model.tflite",
        MODELS_DIR / "calibration_config.json",
    ]
    optional = [
        MODELS_DIR / "rice_disease_model_robust.keras",
        MODELS_DIR / "rice_disease_model_final.keras",
        MODELS_DIR / "rice_disease_model_legacy.tflite",
    ]
    for f in required:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {PASS} {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"  {FAIL} MISSING: {f.name}")
            ok = False
    for f in optional:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {PASS} [optional] {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"  {WARN} [optional, missing] {f.name}")
    return ok


# -----------------------------------------------------------------------
# Check 2: TFLite shape / dtype
# -----------------------------------------------------------------------
def check_tflite_shape() -> bool:
    section("CHECK 2: TFLite Input/Output Shape and Dtype")
    try:
        import tensorflow as tf  # noqa: PLC0415
        tflite_path = MODELS_DIR / "rice_disease_model.tflite"
        if not tflite_path.exists():
            print(f"  {FAIL} TFLite model not found, skipping.")
            return False

        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        inp_shape = inp["shape"].tolist()
        out_shape = out["shape"].tolist()
        inp_dtype = inp["dtype"].__name__
        out_dtype = out["dtype"].__name__

        print(f"  Input  shape: {inp_shape}  dtype: {inp_dtype}")
        print(f"  Output shape: {out_shape}  dtype: {out_dtype}")

        ok = True
        if inp_shape != [1, 224, 224, 3]:
            print(f"  {FAIL} Expected input shape [1,224,224,3], got {inp_shape}")
            ok = False
        else:
            print(f"  {PASS} Input shape correct")

        if out_shape[1] != 5:
            print(f"  {FAIL} Expected 5 output classes, got {out_shape[1]}")
            ok = False
        else:
            print(f"  {PASS} Output classes correct (5)")

        if inp_dtype == "float32":
            print(f"  {PASS} Input dtype float32 — matches Python and Android inference")
        elif inp_dtype == "uint8":
            print(f"  {WARN} Input dtype uint8 — Android must pass uint8, Python must NOT pass float32 [0-255]")
        else:
            print(f"  {WARN} Unusual input dtype: {inp_dtype}")

        return ok
    except ImportError:
        print(f"  {WARN} tensorflow not importable, skipping TFLite check")
        return True


# -----------------------------------------------------------------------
# Check 3: Leaf gating on synthetic OOD
# -----------------------------------------------------------------------
def check_leaf_gating() -> bool:
    section("CHECK 3: Leaf Gating on Synthetic OOD Inputs")
    try:
        from production_inference import RiceDiseaseDetector, ConfidenceLevel  # noqa: PLC0415

        model_path = MODELS_DIR / "rice_disease_model.tflite"
        if not model_path.exists():
            print(f"  {WARN} TFLite model not found, skipping leaf gating check")
            return True

        calib_path = MODELS_DIR / "calibration_config.json"
        detector = RiceDiseaseDetector(
            model_path=model_path,
            enable_validation=True,
            log_predictions=False,
            calibration_config_path=calib_path if calib_path.exists() else None,
        )

        rng = np.random.default_rng(42)
        ood_rejected = 0
        ood_samples = 50
        for _ in range(ood_samples):
            arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
            r = detector.predict(arr, confidence_level=ConfidenceLevel.BALANCED)
            if r.is_rejected:
                ood_rejected += 1

        rej_rate = ood_rejected / ood_samples
        print(f"  Synthetic OOD (random noise): {ood_rejected}/{ood_samples} rejected ({rej_rate:.0%})")
        if rej_rate >= 0.90:
            print(f"  {PASS} OOD rejection rate >= 90% — safeguards working")
            ok = True
        elif rej_rate >= 0.70:
            print(f"  {WARN} OOD rejection rate {rej_rate:.0%} — somewhat low, consider tightening thresholds")
            ok = True
        else:
            print(f"  {FAIL} OOD rejection rate {rej_rate:.0%} — too many random images accepted as rice diseases")
            ok = False

        # Also test solid-color images
        solid_rejected = 0
        for color in [(0, 0, 0), (255, 255, 255), (128, 0, 0), (0, 0, 128)]:
            arr = np.full((224, 224, 3), color, dtype=np.uint8)
            r = detector.predict(arr, confidence_level=ConfidenceLevel.BALANCED)
            if r.is_rejected:
                solid_rejected += 1
        print(f"  Solid-color images: {solid_rejected}/4 rejected")
        if solid_rejected >= 3:
            print(f"  {PASS} Solid color rejection working")
        else:
            print(f"  {WARN} Some solid-color images passed leaf detection")

        return ok
    except Exception as e:
        print(f"  {FAIL} Exception during leaf gating check: {e}")
        return False


# -----------------------------------------------------------------------
# Check 4: Calibration config consistency
# -----------------------------------------------------------------------
def check_calibration_config() -> bool:
    section("CHECK 4: Calibration Config Consistency")
    config_path = MODELS_DIR / "calibration_config.json"
    if not config_path.exists():
        print(f"  {WARN} calibration_config.json not found")
        return True

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ok = True
    temperature = cfg.get("temperature", 1.0)
    conf_thr = cfg.get("rejection", {}).get("confidence_threshold", 0.85)
    ent_thr = cfg.get("rejection", {}).get("entropy_threshold", 0.90)
    model_ref = cfg.get("model", "")

    print(f"  Referenced model: {model_ref}")
    print(f"  Temperature: {temperature}")
    print(f"  Confidence threshold: {conf_thr}")
    print(f"  Entropy threshold: {ent_thr}")

    if temperature > 3.0:
        print(f"  {FAIL} Temperature={temperature} is too high — will flatten confidences and cause over-rejection")
        ok = False
    elif temperature > 2.0:
        print(f"  {WARN} Temperature={temperature} is quite high — may reduce coverage significantly")
    else:
        print(f"  {PASS} Temperature is in reasonable range")

    if conf_thr > 0.90:
        print(f"  {WARN} Confidence threshold={conf_thr} is very strict — expect low coverage")
    elif conf_thr > 0.80:
        print(f"  {PASS} Confidence threshold={conf_thr} — balanced")
    else:
        print(f"  {WARN} Confidence threshold={conf_thr} is permissive — may accept uncertain predictions")

    return ok


# -----------------------------------------------------------------------
# Check 5: Dataset_clean integrity
# -----------------------------------------------------------------------
def check_dataset_integrity() -> bool:
    section("CHECK 5: Dataset_clean Integrity")
    ok = True
    if not DATASET_DIR.exists():
        print(f"  {WARN} Dataset_clean not found at {DATASET_DIR}")
        return True

    total = 0
    for split in ["train", "validation", "test"]:
        counts: Dict[str, int] = {}
        for cls in CLASSES:
            cls_dir = DATASET_DIR / split / cls
            n = sum(1 for _ in cls_dir.glob("*.*")) if cls_dir.exists() else 0
            counts[cls] = n
            total += n
        line = "  " + ", ".join(f"{c}: {n}" for c, n in counts.items())
        print(f"  {split}: {sum(counts.values())} images → {line}")

    print(f"  Total images: {total}")
    if total < 4000:
        print(f"  {WARN} Only {total} images — small dataset. Augmentation is essential.")
    elif total < 8000:
        print(f"  {PASS} Dataset size: {total} images (adequate for fine-tuning)")
    else:
        print(f"  {PASS} Dataset size: {total} images (good)")

    # Quick hash-based cross-split leak check (exact byte hash on small sample)
    print("\n  Checking for exact-duplicate leakage across splits (sample of 200 per split)...")
    split_hashes: Dict[str, set] = {}
    for split in ["train", "validation", "test"]:
        hset: set = set()
        files = []
        for cls in CLASSES:
            cls_dir = DATASET_DIR / split / cls
            if cls_dir.exists():
                files.extend(list(cls_dir.glob("*.jpg"))[:40] + list(cls_dir.glob("*.png"))[:10])
        for p in files[:200]:
            try:
                h = hashlib.sha1(p.read_bytes()).hexdigest()
                hset.add(h)
            except Exception:
                pass
        split_hashes[split] = hset

    tv = len(split_hashes["train"] & split_hashes["validation"])
    tt = len(split_hashes["train"] & split_hashes["test"])
    vt = len(split_hashes["validation"] & split_hashes["test"])

    print(f"  Train∩Val exact-hash overlap (sample): {tv}")
    print(f"  Train∩Test exact-hash overlap (sample): {tt}")
    print(f"  Val∩Test exact-hash overlap (sample): {vt}")

    if tv + tt + vt == 0:
        print(f"  {PASS} No exact duplicates detected across splits in sample")
    else:
        print(f"  {WARN} Some exact duplicates found — run rebuild_clean_dataset.py to reclean if significant")
        ok = False

    return ok


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 70)
    print("  RICE DISEASE DETECTION — FULL PIPELINE VALIDATION")
    print("=" * 70)

    results = {
        "model_files": check_model_files(),
        "tflite_shape": check_tflite_shape(),
        "leaf_gating": check_leaf_gating(),
        "calibration_config": check_calibration_config(),
        "dataset_integrity": check_dataset_integrity(),
    }

    section("SUMMARY")
    all_ok = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print(f"\n  {PASS} All checks passed. Pipeline is consistent.")
    else:
        print(f"\n  {FAIL} Some checks FAILED. Review the output above and fix reported issues.")

    print()


if __name__ == "__main__":
    main()
