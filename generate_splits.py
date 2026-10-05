"""Recreate and check the released 504/177 company split.

Misstatement cases from 2021-2023 are training cases; 2024 cases are test
cases. The original script shuffled the non-misstatement image directory with
seed 42 and took 118 test cases. Directory order can vary across machines;
the included train.txt and test.txt remain the authoritative released lists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {"train": Counter({1: 168, 0: 336}), "test": Counter({1: 59, 0: 118})}
EXPECTED_YEARS = Counter({2021: 65, 2022: 40, 2023: 63, 2024: 59})
FIXED_SPLIT_SHA256 = {
    "train": "258608427d61fb72cee731487f4e73eb4b81c4c9d73610b58750eaeafc36dac8",
    "test": "554f770637919ba1b7bc026fb5664d443529b130f766d18f9f5e7c3a4effa09c",
}


def ordered_fixed_split_codes(path: Path, split: str):
    """Read company codes in the exact order used to save feature arrays."""
    if split not in FIXED_SPLIT_SHA256:
        raise ValueError(f"Unknown split: {split}")
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FIXED_SPLIT_SHA256[split]:
        raise ValueError(f"{path} changed; saved vector row order is no longer verified")
    codes = [Path(line.split()[0]).stem for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(codes) != len(set(codes)):
        raise ValueError(f"Duplicate company code in {path}")
    return codes


def parse_txt_codes(txt_path: str):
    """Read split labels keyed by company code, preserving file order."""
    labels = {}
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            image_path, label = line.strip().split()
            code = os.path.splitext(os.path.basename(image_path))[0]
            labels[code] = int(label)
    return labels


def stock_code(value: str) -> str:
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    if not code.isdigit():
        raise ValueError(f"Invalid stock code: {value!r}")
    return code.zfill(6)


def financial_rows(path: Path):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                rows = list(csv.DictReader(handle))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode {path}")
    required = {"股票代码", "失真标签", "失真年度"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing required columns in {path}: {sorted(required)}")
    by_code = {}
    for row in rows:
        code = stock_code(row["股票代码"])
        if code in by_code:
            raise ValueError(f"Duplicate financial row: {code}")
        by_code[code] = row
    return by_code


def read_split(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image, label = line.split()
            items.append((image, int(label)))
    return items


def make_splits(root: Path):
    by_code = financial_rows(root / "data/standardized_financial_sentiment_features.csv")
    positives = {"train": [], "test": []}
    negatives = []
    years = Counter()
    seen = set()

    for category, label in (("category_1", 1), ("category_2", 0)):
        for filename in os.listdir(root / "JpegImages" / category):
            if not filename.endswith(".png"):
                continue
            image_path = f"JpegImages/{category}/{filename}"
            code = stock_code(Path(filename).stem)
            if code in seen:
                raise ValueError(f"Stock code has multiple images: {code}")
            seen.add(code)
            row = by_code.get(code)
            if row is None or int(float(row["失真标签"])) != label:
                raise ValueError(f"Missing or mismatched financial label: {image_path}")
            if label == 1:
                year = int(float(row["失真年度"]))
                if year not in EXPECTED_YEARS:
                    raise ValueError(f"Unexpected misstatement year: {code} ({year})")
                years[year] += 1
                split = "test" if year == 2024 else "train"
                positives[split].append((image_path, label))
            else:
                negatives.append((image_path, label))

    if seen != set(by_code):
        raise ValueError(f"Financial table and image stock codes differ: {len(seen ^ set(by_code))} codes")
    if years != EXPECTED_YEARS:
        raise ValueError(f"Positive year counts differ: {years}")
    rng = random.Random(42)
    rng.shuffle(negatives)
    splits = {
        "train": positives["train"] + negatives[118:],
        "test": positives["test"] + negatives[:118],
    }
    rng.shuffle(splits["test"])
    rng.shuffle(splits["train"])
    for name, items in splits.items():
        counts = Counter(label for _, label in items)
        if counts != EXPECTED[name]:
            raise ValueError(f"Unexpected {name} counts: {counts}")
    return splits


def verify_released_splits(root: Path, output_dir: Path):
    """Check the supplied lists without depending on directory traversal order."""
    financial = financial_rows(root / "data/standardized_financial_sentiment_features.csv")
    seen = set()
    years = Counter()
    for name in ("train", "test"):
        path = output_dir / f"{name}.txt"
        codes = ordered_fixed_split_codes(path, name)
        rows = read_split(path)
        counts = Counter(label for _, label in rows)
        if counts != EXPECTED[name]:
            raise ValueError(f"Unexpected {name} counts: {counts}")
        for (image, label), code in zip(rows, codes):
            if code in seen:
                raise ValueError(f"Company appears in both splits: {code}")
            seen.add(code)
            if not (root / image).is_file():
                raise FileNotFoundError(root / image)
            row = financial.get(code)
            if row is None or int(float(row["失真标签"])) != label:
                raise ValueError(f"Financial label differs from {name} split: {code}")
            if label == 1:
                year = int(float(row["失真年度"]))
                years[year] += 1
                if (name == "test") != (year == 2024):
                    raise ValueError(f"Positive year in wrong split: {code} ({year})")
    if seen != set(financial) or years != EXPECTED_YEARS:
        raise ValueError("Released split companies or positive years do not match the financial table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the verified split files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "splits")
    args = parser.parse_args()
    paths = [args.output_dir / f"{name}.txt" for name in ("train", "test")]
    if not all(path.exists() for path in paths):
        if not args.write:
            raise FileNotFoundError("Split lists are absent; use --write to create them")
        splits = make_splits(ROOT)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, generated in splits.items():
            path = args.output_dir / f"{name}.txt"
            path.write_text("".join(f"{image} {label}\n" for image, label in generated), encoding="utf-8")
    verify_released_splits(ROOT, args.output_dir)
    print("Split verified: train 504 (168 positive, 336 negative); "
          "test 177 (59 positive, 118 negative).")
    print("Positive samples use misstatement year; negative samples use the original "
          "seeded directory shuffle. Negative sample years are not established by this script.")


if __name__ == "__main__":
    main()
