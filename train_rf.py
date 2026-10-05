"""Fit the RF and its leaf One-Hot encoder using training companies only."""
from __future__ import annotations

import argparse
from pathlib import Path

from rf_features import RFFeatureExtractor, normalize_stock_code, read_csv


def training_labels(path: Path):
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_path, label = line.split()
        code = normalize_stock_code(Path(image_path).stem)
        if code in labels:
            raise ValueError(f"Repeated training company: {code}")
        labels[code] = int(label)
    if not labels:
        raise ValueError("Training split is empty")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_txt", type=Path, default=Path("splits/train.txt"))
    parser.add_argument("--finance", type=Path,
                        default=Path("data/standardized_financial_sentiment_features.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("saved_models/rf"))
    parser.add_argument("--force", action="store_true", help="Refit even if matching artifacts exist")
    args = parser.parse_args()

    labels = training_labels(args.train_txt)
    finance = RFFeatureExtractor._preprocess(read_csv(str(args.finance)))
    train_rows = finance[finance["stock_code"].isin(labels)]
    if len(train_rows) != len(labels):
        raise ValueError(f"Expected {len(labels)} financial training rows; found {len(train_rows)}")
    if any(int(row.label) != labels[row.stock_code] for row in train_rows.itertuples()):
        raise ValueError("Training labels disagree between the split and financial table")

    extractor = RFFeatureExtractor(
        n_estimators=70, max_depth=4, random_state=42,
        class_weight=None, min_samples_leaf=5,
        model_path=str(args.out_dir / "rf_model.joblib"),
        ohe_path=str(args.out_dir / "ohe_encoder.joblib"),
        deterministic=True,
    )
    extractor.prepare(str(args.finance), list(labels), force_retrain=args.force)
    print(f"Fitted on {len(labels)} training companies only; RF leaf One-Hot dimension: {extractor.rf_dim}")
    print(f"RF model: {extractor.model_path}")
    print(f"One-Hot encoder: {extractor.ohe_path}")


if __name__ == "__main__":
    main()
