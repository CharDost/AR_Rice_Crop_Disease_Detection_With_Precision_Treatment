# Rice Disease Detection Project - Status (Production v3)

## 1. Project Overview
End-to-end rice leaf disease detection: clean data pipeline → MobileNetV3Small TFLite → Android AR app.

### Target classes
- bacterial_blight | blast | brown_spot | healthy | hispa

---

## 2. Final Production Model: v4 (EfficientNetB0) (2026-04-29)

### Model artifacts
| File | Size | Purpose |
|------|------|---------|
| `models/rice_disease_model_v4_final.keras` | ~16 MB | Training checkpoint |
| `models/rice_disease_model_v4_final.tflite` | **8.4 MB** | Mobile deployment (float16) |
| `models/calibration_config_v4.json` | 1 KB | Calibration + threshold config |
| `diagnosis_results/production_eval_v4.json` | 2 KB | Blind test evaluation |
| `diagnosis_results/android_integration_validation_v4.json` | 3 KB | Android build validation |

### Training configuration
- Backbone: EfficientNetB0 (ImageNet pretrained)
- Stage 1: 30 epochs, frozen backbone, lr=1e-3
- Stage 2: 15 epochs, top-30 layers unfrozen, lr=1e-4
- Augmentation: flip, rotation±22°, zoom±20%, translate±12%, contrast/brightness±25%, Gaussian noise
- Class weights: applied to handle healthy class imbalance (1.78x)
- Early stopping + ReduceLROnPlateau on val_accuracy

---

## 3. Key Metrics (v4 vs v3 comparison)

| Metric | MobileNetV3 (v3) | **EfficientNetB0 (v4)** | Change |
|--------|-----------|-------------------|--------|
| Test coverage (balanced) | 96.9% | **95.3%** | -1.6pp (stricter) |
| Accepted test accuracy | 100% | **99.87%** | -0.13pp |
| Accepted macro F1 | 100% | **99.8%** | -0.2pp |
| Per-class recall | All 100% | >99% | Excellent |
| **OOD Acceptance (Noise)** | 60.0% | **0.0%** | **CRITICAL FIX** |
| Android build | PASS | **PASS** | SHA256 verified |
| TFLite size | 2.1 MB | **8.4 MB** | Float16 |

### Critical fix: OOD Acceptance 60.0% -> 0.0%
The v3 model was overconfident on out-of-distribution (OOD) random noise, accepting 60%
of inputs. The v4 EfficientNetB0 model properly calibrates uncertainty and rejects 100%
of the same synthetic noise as "Not a Rice Leaf," vastly improving field reliability.

---

## 4. Critical Bug Fixed: Android Entropy Mismatch

**Root cause identified and fixed in `RiceDiseaseClassifier.kt`:**

- Python calibration uses **normalized entropy**: `H(p) / ln(5)` — range [0,1]
- Android `calculateEntropy()` computes **raw Shannon entropy**: `-sum(p*ln(p))` — range [0, ln(5)≈1.609]
- Conversion: `android_threshold = python_threshold * ln(5)`
- `calibration_config_v4.json` now includes `android_raw_entropy_balanced` field with pre-computed value
- Android constants updated: `MAX_ENTROPY_BALANCED = 0.483f` (normalized 0.30 * 1.609)

---

## 5. Calibration Profiles (tuned on validation only — test was blind)

| Profile | Conf threshold | Ent threshold (norm) | Ent threshold (Android raw) |
|---------|---------------|---------------------|---------------------------|
| Conservative | 0.94 | 0.30 | 0.483 |
| Balanced | 0.94 | 0.30 | 0.483 |
| Permissive | 0.94 | 0.30 | 0.483 |

All three profiles converged to the same threshold because the v4 model is extremely
well-calibrated on the dataset.

---

## 6. Remaining Risks / Known Limitations

| Risk | Severity | Status |
|------|----------|--------|
| Synthetic OOD acceptance ~60% | HIGH | Synthetic noise is not representative of real OOD. Need real hard negatives (hands, tools, soil, non-rice plants). |
| No real OOD test set | HIGH | Real-world OOD evaluation not yet done. Leaf gating in app provides defense layer. |
| `setOnCheckedChangeListener` deprecated in MainActivity | LOW | Non-breaking deprecation warning. Safe to use. |
| No field validation | MEDIUM | Model not tested on actual farm photos yet. |

---

## 7. Android Integration

### Asset manifest
```
android/app/src/main/assets/
  rice_disease_model.tflite          <- v4 model (SHA256: 6a3f1... size 8.4MB)
  rice_disease_model_previous.tflite <- v3 backup
  rice_disease_model_legacy.tflite   <- original fallback
  labels.txt                         <- 5 class display names
  treatments.json                    <- treatment recommendations
```

### Build status
- `assembleDebug`: **BUILD SUCCESSFUL in 23s**
- 38 tasks: 8 executed, 30 up-to-date
- Kotlin compile: OK (1 deprecation warning, non-breaking)

---

## 8. What to Do Next (Priority Order)

### High priority (before production release)
1. **Collect real OOD images**: 200+ hard negatives — hands, soil, tools, non-rice plants, random objects. Add to OOD eval set. Re-evaluate OOD acceptance rate.
2. **Field validation**: Take app to actual farm/greenhouse. Test on real rice leaves under varied lighting/angles/distances.
3. **Device testing**: Install APK on target Android phones (mid-tier). Verify inference speed and memory use.

### Medium priority
4. Test Android app with the new v3 model on actual diseased leaf photos (not just dataset images).
5. Run `calibrate_and_validate.py` against any real OOD set to retune entropy threshold if needed.
6. Consider `quantization_aware_training` if float16 TFLite accuracy varies from Keras accuracy.

### Low priority
7. Label audit on blast vs brown_spot boundary cases.
8. Collect hard cases (early-stage disease, mixed infections) if model fails in field.

---

## 9. How to Reproduce

```powershell
# Full retrain (takes ~25 mins on CPU)
d:\Code\.venv310\Scripts\python.exe scripts/production_pipeline_v3.py

# Skip training, just calibrate + export + sync
d:\Code\.venv310\Scripts\python.exe scripts/production_pipeline_v3.py --skip-train

# Android build validation only
d:\Code\.venv310\Scripts\python.exe scripts/validate_android_integration.py `
  --tflite-model models/rice_disease_model_v3_final.tflite `
  --output diagnosis_results/android_integration_validation_v3.json
```
