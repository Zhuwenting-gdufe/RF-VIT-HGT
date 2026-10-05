"""Train an RF + T2T classifier from saved, code-aligned features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch import nn

from rf_features import RFFeatureExtractor, checked_rf_features, normalize_stock_code, read_csv
from generate_splits import ordered_fixed_split_codes


SEED = 42
EPOCHS = 6
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
DROPOUT = 0.6
THRESHOLD = 0.5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_rows(path: Path):
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise ValueError(f"Expected image path and label: {path}")
    codes = [normalize_stock_code(Path(row[0]).stem) for row in rows]
    if len(codes) != len(set(codes)):
        raise ValueError(f"Duplicate company code: {path}")
    return codes, np.asarray([int(row[1]) for row in rows], dtype=np.int64)


def load_features(project: Path):
    saved = project / "saved_models"
    vit_dir = saved / "vit"
    train_codes, y_train = split_rows(project / "splits/train.txt")
    test_codes, y_test = split_rows(project / "splits/test.txt")
    for split, codes in (("train", train_codes), ("test", test_codes)):
        if codes != ordered_fixed_split_codes(project / "splits" / f"{split}.txt", split):
            raise ValueError(f"{split} company order differs from the fixed split")
    if set(train_codes) & set(test_codes):
        raise ValueError("A company occurs in both train and test")
    vectors = {}
    for split, codes in (("train", train_codes), ("test", test_codes)):
        vectors[split] = np.load(vit_dir / f"{split}_vectors.npy")
        if vectors[split].shape != (len(codes), 256):
            raise ValueError(f"Unexpected T2T {split} shape: {vectors[split].shape}")

    finance = RFFeatureExtractor._preprocess(
        read_csv(str(project / "data/standardized_financial_sentiment_features.csv")))
    feature_columns = checked_rf_features(finance)
    if finance["stock_code"].duplicated().any():
        raise ValueError("Duplicate company code in financial table")
    finance = finance.set_index("stock_code")
    all_codes = train_codes + test_codes
    missing = set(all_codes) - set(finance.index)
    if missing:
        raise ValueError(f"Financial data missing companies: {sorted(missing)[:10]}")
    aligned = finance.loc[all_codes]
    if not np.array_equal(aligned["label"].to_numpy().astype(np.int64),
                          np.concatenate([y_train, y_test])):
        raise ValueError("Financial and split labels disagree")

    rf = joblib.load(saved / "rf" / "rf_model.joblib")
    ohe = joblib.load(saved / "rf" / "ohe_encoder.joblib")
    expected = {"n_estimators": 70, "max_depth": 4, "random_state": 42,
                "class_weight": None, "min_samples_leaf": 5}
    if any(rf.get_params()[key] != value for key, value in expected.items()):
        raise ValueError("Saved RF settings differ from this experiment")
    if rf.n_features_in_ != len(feature_columns):
        raise ValueError("Saved RF input dimension differs from the financial feature table")
    x_finance = aligned[feature_columns].to_numpy(dtype=np.float32)
    rf_vectors = ohe.transform(rf.apply(x_finance)).astype(np.float32)
    if rf_vectors.shape != (len(all_codes), 908) or not np.all(rf_vectors.sum(1) == 70):
        raise ValueError(f"Unexpected RF leaf vectors: {rf_vectors.shape}")
    n = len(train_codes)
    x_train = np.concatenate([vectors["train"], rf_vectors[:n]], axis=1).astype(np.float32)
    x_test = np.concatenate([vectors["test"], rf_vectors[n:]], axis=1).astype(np.float32)
    if not (np.isfinite(x_train).all() and np.isfinite(x_test).all()):
        raise ValueError("Non-finite feature value")
    return train_codes, y_train, test_codes, y_test, x_train, x_test, rf, x_finance


def make_classifier():
    return nn.Sequential(
        nn.Linear(1164, 128), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(128, 32), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(32, 2),
    )


@torch.no_grad()
def predict_positive(model, features):
    model.eval()
    logits = model(torch.from_numpy(features))
    return torch.softmax(logits, dim=1)[:, 1].numpy()


def score(y, p):
    pred = (p >= THRESHOLD).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "auc": float(roc_auc_score(y, p))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("saved_models/rf_vit_classifier"))
    parser.add_argument("--vit_metrics_json", type=Path,
                        help="Optional standalone ViT metrics produced by select_vit_epochs.py")
    args = parser.parse_args()
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parent
    out_dir = args.out_dir if args.out_dir.is_absolute() else project / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_codes, y_train, test_codes, y_test, x_train, x_test, rf, x_finance = load_features(project)
    np.save(out_dir / "train_fused_vectors.npy", x_train)
    np.save(out_dir / "test_fused_vectors.npy", x_test)

    seed_everything(SEED)
    model = make_classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Keep the 100-epoch cosine schedule while training the classifier for six epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    features = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} | final batch loss {loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), out_dir / "classifier.pth")

    # Test labels are used only for the following final metric calculation.
    test_probability = predict_positive(model, x_test)
    np.save(out_dir / "test_probabilities.npy", test_probability)
    rf_positive_index = list(rf.classes_).index(1)
    rf_test_probability = rf.predict_proba(x_finance[len(train_codes):])[:, rf_positive_index]
    report = {
        "configuration": {"input_dim": 1164, "t2t_dim": 256, "rf_dim": 908,
                          "hidden_widths": [128, 32], "activation": "ReLU",
                          "dropout": DROPOUT, "layer_norm": False, "rf_gain": 1.0,
                          "optimizer": "AdamW", "learning_rate": LR,
                          "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                          "cosine_schedule_T_max": 100, "seed": SEED,
                          "epochs": EPOCHS, "threshold": THRESHOLD},
        "company_counts": {"train": len(train_codes), "test": len(test_codes)},
        "rf": score(y_test, rf_test_probability),
        "rf_t2t": score(y_test, test_probability),
    }
    if args.vit_metrics_json is not None:
        vit_results = json.loads(args.vit_metrics_json.read_text(encoding="utf-8"))
        report["t2t_vit"] = {key: vit_results[key]
                             for key in ("accuracy", "recall", "f1", "auc")}
    (out_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
