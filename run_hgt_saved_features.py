"""Train HGT with saved RF/T2T vectors; evaluate the test set at the fixed final epoch."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from graph_data_hetero import build_hetero_graph
from hgt_model import HeteroHGT
from generate_splits import ordered_fixed_split_codes, parse_txt_codes


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Preserve the stratified batch order used for HGT training."""

    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.class_indices = {
            cls: np.where(self.labels == cls)[0] for cls in np.unique(self.labels)
        }

    def __iter__(self):
        cls_iters = {
            cls: np.random.permutation(indices).tolist()
            for cls, indices in self.class_indices.items()
        }
        batches = []
        while True:
            batch = []
            empty = True
            for cls, indices in cls_iters.items():
                if not indices:
                    continue
                n_samples = max(
                    1,
                    int(len(self.labels[self.labels == cls]) / len(self.labels) * self.batch_size),
                )
                selected = indices[:n_samples]
                cls_iters[cls] = indices[n_samples:]
                batch.extend(selected)
                empty = False
            if empty:
                break
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            batches.append(batch)
        np.random.shuffle(batches)
        yield from batches

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))


def metrics(truth, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "auc": float(roc_auc_score(truth, probability)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-decay-epochs", type=int, default=100,
                        help="Cosine LR decay length; the 100-epoch baseline uses 100")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--classifier-kind", choices=("deep", "linear", "shallow", "mlp"), default="linear")
    parser.add_argument("--classifier-hidden", type=int, nargs="+", default=[128, 32],
                        help="Hidden widths for the MLP classifier, in order")
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--edge-drop-rate", type=float, default=0.1)
    parser.add_argument("--residual-init", type=float, default=-0.2006706955)
    parser.add_argument("--company-project-width", type=int, default=128)
    parser.add_argument("--other-project-width", type=int, default=128)
    parser.add_argument("--mix-normalization", action="store_true")
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--input-feature-dropout", type=float, default=0.0,
                        help="Mask company input features during HGT training only")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rf-feature-dropout", type=float, default=0.0)
    parser.add_argument("--vit-feature-dropout", type=float, default=0.0)
    parser.add_argument("--rf-modality-dropout", type=float, default=0.0)
    parser.add_argument("--vit-modality-dropout", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-epoch-results", action="store_true",
                        help="Record each epoch's probabilities and score them after training ends")
    parser.add_argument("--save-epoch-checkpoints", action="store_true",
                        help="Save a model checkpoint after every epoch")
    parser.add_argument("--out-dir", default="saved_models/hgt_reproduced")
    args = parser.parse_args()
    if not 0 <= args.input_feature_dropout < 1:
        parser.error("--input-feature-dropout must be in [0, 1)")
    for name in ("rf_feature_dropout", "vit_feature_dropout", "rf_modality_dropout",
                 "vit_modality_dropout"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.weight_decay < 0 or args.input_noise_std < 0:
        parser.error("Weight decay and input noise must be nonnegative")
    if args.lr_decay_epochs < 1:
        parser.error("--lr-decay-epochs must be positive")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vector_dir = Path("saved_models/rf_vit_classifier")
    code_to_rf, code_to_visual = {}, {}
    for split in ("train", "test"):
        codes = ordered_fixed_split_codes(Path("splits") / f"{split}.txt", split)
        vectors = np.load(vector_dir / f"{split}_fused_vectors.npy")
        if vectors.shape != (len(codes), 1164) or len(set(codes)) != len(codes):
            raise ValueError(f"Invalid saved {split} vectors or split company codes")
        for code, vector in zip(codes, vectors):
            code_to_visual[code] = vector[:256].copy()
            code_to_rf[code] = vector[256:].copy()

    train_labels = parse_txt_codes("splits/train.txt")
    if list(train_labels) != ordered_fixed_split_codes(Path("splits/train.txt"), "train"):
        raise ValueError("Training labels do not match the fixed split order")
    data, meta = build_hetero_graph(
        "data/concepts.csv", "data/executives.csv", "data/shareholders.csv",
        "data/subsidiaries.csv", code_to_rf, code_to_visual, train_labels,
    )
    company_index = meta["lid"]["company"]
    train_idx = torch.tensor([company_index[code] for code in train_labels])
    y = data["company"].y
    sampler = StratifiedBatchSampler(y[train_idx].numpy(), args.batch_size)
    in_dims = {kind: data[kind].x.shape[1] for kind in data.node_types}
    model = HeteroHGT(data.metadata(), in_dims, hidden_dim=128, num_layers=2,
                      dropout=args.dropout, heads=args.heads,
                      edge_drop_rate=args.edge_drop_rate,
                      classifier_kind=args.classifier_kind,
                      residual_init=args.residual_init,
                      classifier_hidden_dims=args.classifier_hidden,
                      company_project_width=args.company_project_width,
                      other_project_width=args.other_project_width,
                      mix_normalization=args.mix_normalization,
                      projection_dropout=args.projection_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=args.weight_decay)
    decay_epochs = args.lr_decay_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=1e-6,
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    alpha_history = []
    epoch_probabilities = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in sampler:
            optimizer.zero_grad()
            indices = train_idx[torch.as_tensor(batch, dtype=torch.long)]
            has_input_augmentation = any((args.input_feature_dropout,
                                          args.rf_feature_dropout, args.vit_feature_dropout,
                                          args.rf_modality_dropout, args.vit_modality_dropout,
                                          args.input_noise_std))
            if has_input_augmentation:
                company_input = data["company"].x.clone()
                if args.input_feature_dropout:
                    company_input = F.dropout(company_input, p=args.input_feature_dropout,
                                              training=True)
                if args.rf_feature_dropout:
                    company_input[:, :908] = F.dropout(company_input[:, :908],
                                                        p=args.rf_feature_dropout, training=True)
                if args.vit_feature_dropout:
                    company_input[:, 908:] = F.dropout(company_input[:, 908:],
                                                        p=args.vit_feature_dropout, training=True)
                if args.rf_modality_dropout:
                    mask = (torch.rand(company_input.size(0), 1) >= args.rf_modality_dropout)
                    company_input[:, :908] *= mask / (1 - args.rf_modality_dropout)
                if args.vit_modality_dropout:
                    mask = (torch.rand(company_input.size(0), 1) >= args.vit_modality_dropout)
                    company_input[:, 908:] *= mask / (1 - args.vit_modality_dropout)
                if args.input_noise_std:
                    company_input += torch.randn_like(company_input) * args.input_noise_std
                logits = model(data, x_override={"company": company_input})[indices]
            else:
                logits = model(data)[indices]
            loss = torch.nn.functional.cross_entropy(logits, y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        # Stop stepping after T_max; CosineAnnealingLR would otherwise rise again.
        if epoch <= decay_epochs:
            scheduler.step()
        alpha_history.append({"epoch": epoch, "graph_feature_weight": model.get_alpha()})
        if args.record_epoch_results:
            model.eval()
            with torch.no_grad():
                epoch_probabilities.append(model(data).softmax(dim=-1)[:, 1].numpy().copy())
        if args.save_epoch_checkpoints:
            torch.save(model.state_dict(), output / f"checkpoint_epoch{epoch}.pth")
        print(f"epoch={epoch} train_loss={np.mean(losses):.5f} alpha={model.get_alpha():.4f} "
              f"elapsed_seconds={time.monotonic()-started:.1f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, (company_initial, graph_output, company_final) = model(
            data, return_branches=True)
        probability = logits.softmax(dim=-1)[:, 1].numpy()
    train_truth = np.array([train_labels[code] for code in train_labels])
    train_result = metrics(train_truth, probability[train_idx.numpy()], args.threshold)

    # Read test labels only after the fixed final training epoch is complete.
    test_labels = parse_txt_codes("splits/test.txt")
    if list(test_labels) != ordered_fixed_split_codes(Path("splits/test.txt"), "test"):
        raise ValueError("Test labels do not match the fixed split order")
    test_idx = np.array([company_index[code] for code in test_labels])
    test_truth = np.array([test_labels[code] for code in test_labels])
    test_result = metrics(test_truth, probability[test_idx], args.threshold)
    with torch.no_grad():
        initial_for_mix = (F.layer_norm(company_initial, (model.hidden_dim,))
                           if args.mix_normalization else company_initial)
        graph_for_mix = (F.layer_norm(graph_output, (model.hidden_dim,))
                         if args.mix_normalization else graph_output)
        alpha = model.get_alpha()
        norm_median = lambda value: float(value[test_idx].norm(dim=1).median())
        branch_norms = {
            "initial_raw": norm_median(company_initial),
            "graph_raw": norm_median(graph_output),
            "initial_weighted": norm_median((1 - alpha) * initial_for_mix),
            "graph_weighted": norm_median(alpha * graph_for_mix),
            "final": norm_median(company_final),
        }

    epoch_results = []
    for epoch, epoch_probability in enumerate(epoch_probabilities, start=1):
        epoch_results.append({
            "epoch": epoch,
            "train": metrics(train_truth, epoch_probability[train_idx.numpy()], args.threshold),
            "test": metrics(test_truth, epoch_probability[test_idx], args.threshold),
        })

    torch.save(model.state_dict(), output / "final_model.pth")
    report = {"epochs": args.epochs, "heads": args.heads, "batch_size": args.batch_size,
              "lr_decay_epochs": decay_epochs,
              "optimizer": "AdamW", "learning_rate": 3e-4,
              "learning_rate_scheduler": "cosine_annealing_then_hold",
              "minimum_learning_rate": 1e-6, "gradient_clip_norm": 1.0,
              "loss": "cross_entropy", "hidden_dim": 128, "num_layers": 2,
              "seed": args.seed, "classifier_kind": args.classifier_kind,
              "classifier_hidden_dims": args.classifier_hidden if args.classifier_kind == "mlp" else None,
              "dropout": args.dropout, "edge_drop_rate": args.edge_drop_rate,
              "company_project_width": args.company_project_width,
              "other_project_width": args.other_project_width,
              "mix_normalization": args.mix_normalization,
              "projection_dropout": args.projection_dropout,
              "input_feature_dropout": args.input_feature_dropout,
              "weight_decay": args.weight_decay,
              "rf_feature_dropout": args.rf_feature_dropout,
              "vit_feature_dropout": args.vit_feature_dropout,
              "rf_modality_dropout": args.rf_modality_dropout,
              "vit_modality_dropout": args.vit_modality_dropout,
              "input_noise_std": args.input_noise_std,
              "save_epoch_checkpoints": args.save_epoch_checkpoints,
              "test_branch_norm_medians": branch_norms,
              "residual_init": args.residual_init,
              "initial_graph_feature_weight": float(torch.sigmoid(torch.tensor(args.residual_init))),
              "final_graph_feature_weight": model.get_alpha(),
              "alpha_history": alpha_history,
              "threshold": args.threshold,
              "rf_dimension": 908, "vit_dimension": 256,
              "company_input_dimension": 1164, "other_node_input_dimension": 512,
              "train_split_file": "splits/train.txt", "test_split_file": "splits/test.txt",
              "fused_vector_directory": str(vector_dir),
              "semantic_cache_directory": "cache",
              "company_order": "RF then T2T-ViT", "train_count": len(train_labels),
              "test_count": len(test_labels), "train": train_result, "test": test_result,
              "final_model_file": str(output / "final_model.pth"),
              "elapsed_seconds": time.monotonic() - started}
    if args.record_epoch_results:
        report["epoch_results"] = epoch_results
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
