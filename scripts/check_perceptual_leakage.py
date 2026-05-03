
import imagehash
from PIL import Image
from pathlib import Path

train_hashes = set()
val_leak_count = 0
total_val = 0

print("Hashing Train Images...")
for p in Path("Dataset_clean/train").rglob("*.*"):
    if p.is_file():
        try:
            h = imagehash.phash(Image.open(p))
            train_hashes.add(str(h))
        except:
            pass

print("Checking Validation Images against Train Hashes...")
for p in Path("Dataset_clean/validation").rglob("*.*"):
    if p.is_file():
        total_val += 1
        try:
            h = str(imagehash.phash(Image.open(p)))
            if h in train_hashes:
                val_leak_count += 1
        except:
            pass

print(f"\nTotal Validation Images: {total_val}")
print(f"Perceptually Leaked from Train: {val_leak_count} ({(val_leak_count/max(1, total_val))*100:.2f}%)")

