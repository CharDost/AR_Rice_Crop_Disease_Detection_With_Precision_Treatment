"""
Evaluate production inference safeguards (confidence + entropy + leaf validation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from production_inference import RiceDiseaseDetector, ConfidenceLevel


CLASS_MAP = {
    "bacterial_blight": "Bacterial Blight",
    "blast": "Blast",
    "brown_spot": "Brown Spot",
    "healthy": "Healthy",
    "hispa": "Hispa",
}
REV_MAP = {v: k for k, v in CLASS_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference safeguards")
    parser.add_argument("--model", type=Path, default=Path("models/rice_disease_model_final.keras"))
    parser.add_argument("--calibration", type=Path, default=Path("models/calibration_config.json"))
    parser.add_argument("--test-dir", type=Path, default=Path("Dataset_clean/test"))
    parser.add_argument("--output", type=Path, default=Path("diagnosis_results/safeguard_eval.json"))
    parser.add_argument("--ood-dir", type=Path, default=Path("OOD"))
    parser.add_argument("--ood-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_test_images(test_dir: Path):
    for klass in CLASS_MAP.keys():
        cdir = test_dir / klass
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob("*.*")):
            yield p, klass


def main() -> None:
    args = parse_args()

    det = RiceDiseaseDetector(
        model_path=args.model,
        calibration_config_path=args.calibration,
        enable_validation=True,
        log_predictions=False,
    )

    y_true: List[str] = []
    y_pred: List[str] = []
    accepted = 0
    rejected = 0

    for path, klass in iter_test_images(args.test_dir):
        r = det.predict(str(path), confidence_level=ConfidenceLevel.BALANCED)
        y_true.append(klass)
        if r.is_rejected:
            y_pred.append("unknown")
            rejected += 1
        else:
            y_pred.append(REV_MAP.get(r.predicted_class, "unknown"))
            accepted += 1

    # Metrics on accepted only
    idx = [i for i, p in enumerate(y_pred) if p != "unknown"]
    if idx:
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        acc = float(accuracy_score(yt, yp))
        p, r, f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            labels=list(CLASS_MAP.keys()),
            average="macro",
            zero_division=0,
        )
    else:
        acc, p, r, f1 = 0.0, 0.0, 0.0, 0.0

    coverage = accepted / max(1, len(y_true))

    # Synthetic OOD test
    rng = np.random.default_rng(args.seed)
    ood_rejected = 0
    ood_conf = []
    for _ in range(args.ood_n):
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        r_ood = det.predict(arr, confidence_level=ConfidenceLevel.BALANCED)
        ood_conf.append(float(r_ood.confidence))
        if r_ood.is_rejected:
            ood_rejected += 1

    # Real OOD test
    real_ood_rejected = 0
    real_ood_conf = []
    real_ood_count = 0
    if args.ood_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"):
            for file_path in args.ood_dir.rglob(ext):
                try:
                    r_ood = det.predict(str(file_path), confidence_level=ConfidenceLevel.BALANCED)
                    real_ood_conf.append(float(r_ood.confidence))
                    real_ood_count += 1
                    if r_ood.is_rejected:
                        real_ood_rejected += 1
                except Exception:
                    pass

    payload = {
        "test": {
            "samples": len(y_true),
            "accepted": accepted,
            "rejected": rejected,
            "coverage": coverage,
            "accepted_accuracy": acc,
            "accepted_macro_precision": float(p),
            "accepted_macro_recall": float(r),
            "accepted_macro_f1": float(f1),
        },
        "synthetic_ood": {
            "samples": args.ood_n,
            "rejected": ood_rejected,
            "rejection_rate": ood_rejected / max(1, args.ood_n),
            "mean_confidence": float(np.mean(ood_conf)),
            "p95_confidence": float(np.percentile(ood_conf, 95)),
        },
    }
    
    if real_ood_count > 0:
        payload["real_ood"] = {
            "samples": real_ood_count,
            "rejected": real_ood_rejected,
            "rejection_rate": real_ood_rejected / real_ood_count,
            "mean_confidence": float(np.mean(real_ood_conf)),
            "p95_confidence": float(np.percentile(real_ood_conf, 95)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
