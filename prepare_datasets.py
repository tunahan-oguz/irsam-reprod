#!/usr/bin/env python3
"""
Prepare datasets for IRSAM training/evaluation.

Reorganizes downloaded datasets into the folder structure expected by demo.py:
  datasets/<name>/test_images/
  datasets/<name>/test_masks/
  datasets/<name>/trainval_images/
  datasets/<name>/trainval_masks/

Handles:
- IRSTD-1k: source has images/ and labels/ (renamed to masks), names match
- NUDT-SIRST00: source has images/ and masks/, names match, test.txt includes .png extension
- Sirstv2_512: source has images/ and masks/, but mask names have '_pixels0' suffix 
  (e.g., image: Misc_1.png, mask: Misc_1_pixels0.png). Masks are renamed to match images.
"""

import os
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).parent / "datasets"


def read_split_file(filepath):
    """Read a split file (test.txt or trainval.txt) and return list of basenames (no extension)."""
    names = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Remove .png extension if present (NUDT test.txt includes it)
                if line.endswith('.png'):
                    line = line[:-4]
                names.append(line)
    return names


def prepare_irstd1k():
    """
    IRSTD-1k:
    - Source: images/ (XDU0.png, ...) and labels/ (XDU0.png, ...)
    - test.txt / trainval.txt entries: XDU514, XDU646, ... (no extension)
    - Target: test_images/, test_masks/, trainval_images/, trainval_masks/
    """
    dataset_dir = BASE_DIR / "IRSTD-1k"
    src_images = dataset_dir / "images"
    src_labels = dataset_dir / "labels"

    if not src_images.exists():
        print(f"[IRSTD-1k] SKIP: {src_images} not found (already processed?)")
        return

    test_names = read_split_file(dataset_dir / "test.txt")
    trainval_names = read_split_file(dataset_dir / "trainval.txt")

    print(f"[IRSTD-1k] test: {len(test_names)}, trainval: {len(trainval_names)}")

    # Create target directories
    for d in ["test_images", "test_masks", "trainval_images", "trainval_masks"]:
        (dataset_dir / d).mkdir(exist_ok=True)

    # Copy test split
    for name in test_names:
        img_src = src_images / f"{name}.png"
        lbl_src = src_labels / f"{name}.png"
        if img_src.exists() and lbl_src.exists():
            shutil.copy2(img_src, dataset_dir / "test_images" / f"{name}.png")
            shutil.copy2(lbl_src, dataset_dir / "test_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: {name}")

    # Copy trainval split
    for name in trainval_names:
        img_src = src_images / f"{name}.png"
        lbl_src = src_labels / f"{name}.png"
        if img_src.exists() and lbl_src.exists():
            shutil.copy2(img_src, dataset_dir / "trainval_images" / f"{name}.png")
            shutil.copy2(lbl_src, dataset_dir / "trainval_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: {name}")

    copied_test = len(list((dataset_dir / "test_images").iterdir()))
    copied_train = len(list((dataset_dir / "trainval_images").iterdir()))
    print(f"[IRSTD-1k] Done: {copied_test} test, {copied_train} trainval images copied")


def prepare_nudt_sirst():
    """
    NUDT-SIRST00:
    - Source: images/ (000001.png, ...) and masks/ (000001.png, ...)
    - test.txt entries: 001000.png, ... (WITH .png extension)
    - trainval.txt entries: same format
    - Target: test_images/, test_masks/, trainval_images/, trainval_masks/
    """
    dataset_dir = BASE_DIR / "NUDT-SIRST00"
    src_images = dataset_dir / "images"
    src_masks = dataset_dir / "masks"

    if not src_images.exists():
        print(f"[NUDT-SIRST00] SKIP: {src_images} not found (already processed?)")
        return

    test_names = read_split_file(dataset_dir / "test.txt")
    trainval_names = read_split_file(dataset_dir / "trainval.txt")

    print(f"[NUDT-SIRST00] test: {len(test_names)}, trainval: {len(trainval_names)}")

    # Create target directories
    for d in ["test_images", "test_masks", "trainval_images", "trainval_masks"]:
        (dataset_dir / d).mkdir(exist_ok=True)

    # Copy test split
    for name in test_names:
        img_src = src_images / f"{name}.png"
        msk_src = src_masks / f"{name}.png"
        if img_src.exists() and msk_src.exists():
            shutil.copy2(img_src, dataset_dir / "test_images" / f"{name}.png")
            shutil.copy2(msk_src, dataset_dir / "test_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: {name}")

    # Copy trainval split
    for name in trainval_names:
        img_src = src_images / f"{name}.png"
        msk_src = src_masks / f"{name}.png"
        if img_src.exists() and msk_src.exists():
            shutil.copy2(img_src, dataset_dir / "trainval_images" / f"{name}.png")
            shutil.copy2(msk_src, dataset_dir / "trainval_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: {name}")

    copied_test = len(list((dataset_dir / "test_images").iterdir()))
    copied_train = len(list((dataset_dir / "trainval_images").iterdir()))
    print(f"[NUDT-SIRST00] Done: {copied_test} test, {copied_train} trainval images copied")


def prepare_sirstv2():
    """
    Sirstv2_512:
    - Source: images/ (Misc_1.png, ...) and masks/ (Misc_1_pixels0.png, ...)
    - test.txt / trainval.txt entries: Misc_70, Misc_214, ... (no extension)
    - CRITICAL: mask filenames have '_pixels0' suffix that must be removed
      so the dataloader can derive mask path from image path.
    - Target: test_images/, test_masks/, trainval_images/, trainval_masks/
    """
    dataset_dir = BASE_DIR / "Sirstv2_512"
    src_images = dataset_dir / "images"
    src_masks = dataset_dir / "masks"

    if not src_images.exists():
        print(f"[Sirstv2_512] SKIP: {src_images} not found (already processed?)")
        return

    test_names = read_split_file(dataset_dir / "test.txt")
    trainval_names = read_split_file(dataset_dir / "trainval.txt")

    print(f"[Sirstv2_512] test: {len(test_names)}, trainval: {len(trainval_names)}")

    # Create target directories
    for d in ["test_images", "test_masks", "trainval_images", "trainval_masks"]:
        (dataset_dir / d).mkdir(exist_ok=True)

    # Copy test split
    for name in test_names:
        img_src = src_images / f"{name}.png"
        # Mask has '_pixels0' suffix in original dataset
        msk_src = src_masks / f"{name}_pixels0.png"
        if img_src.exists() and msk_src.exists():
            shutil.copy2(img_src, dataset_dir / "test_images" / f"{name}.png")
            # Rename mask to match image name (remove _pixels0 suffix)
            shutil.copy2(msk_src, dataset_dir / "test_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: img={img_src.exists()}, msk={msk_src.exists()} for {name}")

    # Copy trainval split
    for name in trainval_names:
        img_src = src_images / f"{name}.png"
        msk_src = src_masks / f"{name}_pixels0.png"
        if img_src.exists() and msk_src.exists():
            shutil.copy2(img_src, dataset_dir / "trainval_images" / f"{name}.png")
            shutil.copy2(msk_src, dataset_dir / "trainval_masks" / f"{name}.png")
        else:
            print(f"  [WARN] Missing: img={img_src.exists()}, msk={msk_src.exists()} for {name}")

    copied_test = len(list((dataset_dir / "test_images").iterdir()))
    copied_train = len(list((dataset_dir / "trainval_images").iterdir()))
    print(f"[Sirstv2_512] Done: {copied_test} test, {copied_train} trainval images copied")


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing datasets for IRSAM")
    print("=" * 60)

    prepare_irstd1k()
    print()
    prepare_nudt_sirst()
    print()
    prepare_sirstv2()

    print()
    print("=" * 60)
    print("All datasets prepared!")
    print("You can now safely remove the original images/, labels/, masks/ folders if desired.")
    print("=" * 60)
