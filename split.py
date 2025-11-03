
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

# --- Load config ---
with open("C:/Users/nroy/Desktop/Patch_FM_Classifier/data/config.yaml", "r") as f:
    config = yaml.safe_load(f)

src_dir = Path(config["data_dir"])
dst_dir = Path(config["output_dir"])
classes = config["classes"]
test_size = config["splits"]["test_size"]
val_size = config["splits"]["val_size"]
random_state = config["splits"]["random_state"]

# --- Create folders ---
for split in ["train", "val", "test"]:
    for cls in classes:
        (dst_dir / split / cls).mkdir(parents=True, exist_ok=True)

# --- Split and copy ---
for cls in classes:
    files = list((src_dir / cls).glob("*.tif"))
    trainval, test = train_test_split(files, test_size=test_size, random_state=random_state)
    train, val = train_test_split(trainval, test_size=val_size, random_state=random_state)
    print(f"\nProcessing class: {cls}")

    for f in tqdm(train, desc=f"Copying train/{cls}", unit="file"):
        shutil.copy(f, dst_dir / "train" / cls / f.name)

    for f in tqdm(val, desc=f"Copying val/{cls}", unit="file"):
        shutil.copy(f, dst_dir / "val" / cls / f.name)

    for f in tqdm(test, desc=f"Copying test/{cls}", unit="file"):
        shutil.copy(f, dst_dir / "test" / cls / f.name)