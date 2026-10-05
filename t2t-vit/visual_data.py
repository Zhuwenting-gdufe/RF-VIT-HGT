"""Image dataset, seeding, and validation metrics shared by ViT stages."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset

from rf_features import normalize_stock_code


class CandlestickDataset(Dataset):
    def __init__(self, split_file, root_dir, rf_features, transform):
        self.root_dir = root_dir
        self.rf_features = rf_features
        self.transform = transform
        self.samples = []
        with open(split_file, encoding="utf-8") as handle:
            for line in handle:
                image_path, label = line.strip().split()
                code = normalize_stock_code(os.path.splitext(os.path.basename(image_path))[0])
                self.samples.append((image_path, code, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, code, label = self.samples[index]
        image = Image.open(os.path.join(self.root_dir, image_path)).convert("RGB")
        image = self.transform(image)
        rf_vector = (torch.from_numpy(self.rf_features[code]).float()
                     if self.rf_features is not None else torch.empty(0))
        return image, rf_vector, label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    labels, predictions, probabilities = [], [], []
    for images, rf_vectors, targets in loader:
        images = images.to(device)
        rf_vectors = rf_vectors.to(device)
        logits = classifier(torch.cat([encoder.forward_features(images), rf_vectors], dim=1))
        positive_probabilities = torch.softmax(logits, dim=1)[:, 1]
        labels.extend(targets.numpy().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        probabilities.extend(positive_probabilities.cpu().numpy().tolist())
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan")
    return accuracy, f1, auc
