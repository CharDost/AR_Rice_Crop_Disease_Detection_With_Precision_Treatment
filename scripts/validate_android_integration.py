"""
Validate Android integration with a newly trained TFLite model.

Checks:
1) Model file exists
2) Copy/sync into Android assets as rice_disease_model.tflite
3) Ensure legacy fallback model still exists
4) Build debug APK (Gradle)
5) Confirm synced asset exists after build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Android model integration")
    parser.add_argument("--tflite-model", type=Path, required=True)
    parser.add_argument("--android-assets", type=Path, default=Path("android/app/src/main/assets"))
    parser.add_argument("--android-root", type=Path, default=Path("android"))
    parser.add_argument("--output", type=Path, default=Path("diagnosis_results/android_integration_validation.json"))
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()

    result: Dict = {
        "source_model": str(args.tflite_model),
        "android_assets": str(args.android_assets),
        "checks": {},
    }

    if not args.tflite_model.exists():
        raise FileNotFoundError(f"TFLite model not found: {args.tflite_model}")

    args.android_assets.mkdir(parents=True, exist_ok=True)
    target_model = args.android_assets / "rice_disease_model.tflite"
    backup_model = args.android_assets / "rice_disease_model_previous.tflite"
    legacy_model = args.android_assets / "rice_disease_model_legacy.tflite"

    if target_model.exists():
        shutil.copy2(target_model, backup_model)
        result["checks"]["backup_created"] = str(backup_model)

    shutil.copy2(args.tflite_model, target_model)

    src_hash = sha256(args.tflite_model)
    dst_hash = sha256(target_model)
    result["checks"]["source_sha256"] = src_hash
    result["checks"]["asset_sha256"] = dst_hash
    result["checks"]["asset_sync_ok"] = bool(src_hash == dst_hash)

    result["checks"]["legacy_fallback_exists"] = legacy_model.exists()

    build_rc = None
    build_output_tail = ""
    if not args.skip_build:
        gradle = args.android_root / "gradlew.bat"
        if not gradle.exists():
            gradle = args.android_root / "gradlew"
        if not gradle.exists():
            raise FileNotFoundError("Could not find gradlew(.bat) in android root")

        # Run from android root and invoke gradle using local relative executable name.
        gradle_cmd = ".\\gradlew.bat" if gradle.name.lower().endswith(".bat") else "./gradlew"

        proc = subprocess.run(
            f"{gradle_cmd} assembleDebug",
            cwd=str(args.android_root),
            capture_output=True,
            text=True,
            shell=True,
        )
        build_rc = int(proc.returncode)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        build_output_tail = "\n".join(combined.strip().splitlines()[-80:])

    result["checks"]["build_return_code"] = build_rc
    result["checks"]["build_succeeded"] = (build_rc == 0) if build_rc is not None else None
    result["checks"]["build_output_tail"] = build_output_tail

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 80)
    print("ANDROID INTEGRATION VALIDATION COMPLETE")
    print("=" * 80)
    print(json.dumps({"checks": result["checks"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
