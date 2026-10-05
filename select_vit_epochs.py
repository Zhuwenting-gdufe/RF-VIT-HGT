"""Select an image-only ViT training length using training-set CV only.

Run five independent --fold jobs, select the epoch with the highest mean
validation AUC, then run --selected_epoch once to evaluate the test set.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from rf_features import normalize_stock_code

t2t_vit_kline = import_module("t2t-vit.t2t_vit_kline").t2t_vit_kline
_visual_data = import_module("t2t-vit.visual_data")
CandlestickDataset = _visual_data.CandlestickDataset
evaluate = _visual_data.evaluate
seed_everything = _visual_data.seed_everything


def make_model(device):
    encoder = t2t_vit_kline(
        pretrained=False, num_classes=2, embed_dim=256, depth=4, drop_rate=0.2,
    ).to(device)
    classifier = nn.Linear(256, 2).to(device)
    return encoder, classifier


def fit(encoder, classifier, loader, optimizer, device):
    encoder.train()
    classifier.train()
    total = 0.0
    for images, _, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = classifier(encoder.forward_features(images))
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_final(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    labels, predictions, probabilities = [], [], []
    for images, _, targets in loader:
        logits = classifier(encoder.forward_features(images.to(device)))
        labels.extend(targets.numpy().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return (accuracy_score(labels, predictions), recall_score(labels, predictions),
            f1_score(labels, predictions), roc_auc_score(labels, probabilities))


class ImagesForFeatures(Dataset):
    """Read image paths only; labels are not needed to extract vectors."""

    def __init__(self, split_file, root_dir, transform):
        self.paths = [line.split()[0] for line in Path(split_file).read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        self.codes = [normalize_stock_code(Path(path).stem) for path in self.paths]
        if len(self.codes) != len(set(self.codes)):
            raise ValueError(f"Duplicate company code in {split_file}")
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root_dir / self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def extract_vectors(encoder, split_file, img_root, transform, batch_size, device):
    dataset = ImagesForFeatures(split_file, img_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    parts = [encoder.forward_features(images.to(device)).cpu().numpy() for images in loader]
    vectors = np.concatenate(parts).astype(np.float32)
    if vectors.shape != (len(dataset), 256):
        raise ValueError(f"Unexpected feature shape for {split_file}: {vectors.shape}")
    return vectors, dataset.codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, help="Zero-based fold index, 0 through 4")
    group.add_argument("--select_cv", type=Path, help="Directory containing fold0.json through fold4.json")
    group.add_argument("--selected_epoch", type=int, help="Retrain on all training cases and evaluate test once")
    parser.add_argument("--train_txt", default="splits/train.txt")
    parser.add_argument("--test_txt", default="splits/test.txt")
    parser.add_argument("--img_root", default=".")
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--save_dir", type=Path, help="Save the final encoder and its feature vectors here")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.fold is not None and not 0 <= args.fold < 5:
        parser.error("--fold must be 0, 1, 2, 3, or 4")
    if args.selected_epoch is not None and not 1 <= args.selected_epoch <= args.epochs:
        parser.error("--selected_epoch must be between 1 and --epochs")
    if args.fold is not None and args.output_json is None:
        parser.error("--output_json is required for a fold")
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.select_cv is not None:
        folds = [json.loads((args.select_cv / f"fold{i}.json").read_text(encoding="utf-8"))
                 for i in range(5)]
        if any(fold["fold"] != i or len(fold["history"]) != args.epochs
               for i, fold in enumerate(folds)):
            parser.error("CV files must cover folds 0 through 4 with the requested epoch count")
        auc = np.array([[row["auc"] for row in fold["history"]] for fold in folds])
        mean_auc = auc.mean(axis=0)
        chosen = int(np.argmax(mean_auc) + 1)
        selection = {"selected_epoch": chosen, "mean_validation_auc": float(mean_auc[chosen - 1]),
                     "mean_auc_by_epoch": mean_auc.tolist(),
                     "fold_best_epochs": [int(np.argmax(row) + 1) for row in auc]}
        (args.select_cv / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8")
        print(f"SELECTED EPOCH {chosen} | mean_validation_auc {mean_auc[chosen-1]:.4f}")
        return

    seed_everything(args.seed + (args.fold or 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.0), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    train_data = CandlestickDataset(args.train_txt, args.img_root, None, train_transform)
    labels = np.array([label for _, _, label in train_data.samples])
    if args.fold is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        train_idx, val_idx = list(cv.split(np.zeros(len(labels)), labels))[args.fold]
        val_data = CandlestickDataset(args.train_txt, args.img_root, None, eval_transform)
        train_loader = DataLoader(Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(val_data, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
        total_epochs = args.epochs
        print(f"fold={args.fold} train={len(train_idx)} validation={len(val_idx)} device={device}", flush=True)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        total_epochs = args.selected_epoch
        print(f"final training={len(train_data)} selected_epoch={total_epochs} device={device}", flush=True)

    encoder, classifier = make_model(device)
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()), lr=3e-4, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_fold_auc = float("-inf")
    best_fold_epoch = None
    for epoch in range(1, total_epochs + 1):
        train_loss = fit(encoder, classifier, train_loader, optimizer, device)
        scheduler.step()
        if args.fold is not None:
            acc, f1, auc = evaluate(encoder, classifier, val_loader, device)
            history.append({"epoch": epoch, "accuracy": acc, "f1": f1, "auc": auc})
            if auc > best_fold_auc:
                best_fold_auc, best_fold_epoch = auc, epoch
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {acc:.4f} | val_f1 {f1:.4f} | val_auc {auc:.4f}", flush=True)
        else:
            print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f}", flush=True)

    if args.fold is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"fold": args.fold, "seed": args.seed,
            "train_count": len(train_idx), "validation_count": len(val_idx),
            "best_epoch": best_fold_epoch, "history": history}, indent=2), encoding="utf-8")
        print(f"Saved validation history: {args.output_json}", flush=True)
    else:
        if args.save_dir is not None:
            torch.save(encoder.state_dict(), args.save_dir / "full_encoder.pth")
            for name, split_file in (("train", args.train_txt), ("test", args.test_txt)):
                vectors, _ = extract_vectors(
                    encoder, split_file, args.img_root, eval_transform, args.batch_size, device)
                np.save(args.save_dir / f"{name}_vectors.npy", vectors)
                print(f"Saved {name} vectors: {vectors.shape}", flush=True)
        # The test list is opened only after the epoch has been fixed and full training is complete.
        test_data = CandlestickDataset(args.test_txt, args.img_root, None, eval_transform)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        acc, recall, f1, auc = evaluate_final(encoder, classifier, test_loader, device)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps({
                "selected_epoch": total_epochs, "accuracy": acc, "recall": recall,
                "f1": f1, "auc": auc, "train_count": len(train_data),
                "test_count": len(test_data), "seed": args.seed,
            }, indent=2), encoding="utf-8")
        print(f"TEST ONCE | epoch {total_epochs} | accuracy {acc:.4f} | recall {recall:.4f} | f1 {f1:.4f} | auc {auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
