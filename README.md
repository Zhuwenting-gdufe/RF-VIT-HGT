# RF-ViT-HGT

This repository contains the RF, T2T-ViT, and heterogeneous graph transformer (HGT) code, fixed data splits, candlestick chart images, relationship tables, and reusable intermediate features. Run all commands from the repository root.

## 1. Environment and data

Use Python 3.8.20. Other dependency versions are listed in `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate rf-vit-hgt
python generate_splits.py
```

`data/` contains the financial and sentiment features and the relationship tables. `JpegImages/` contains the candlestick charts. Each line of `splits/train.txt` and `splits/test.txt` gives a relative image path and its label. The fixed training split contains 504 companies (168 positive and 336 negative); the test split contains 177 companies (59 positive and 118 negative). In `generate_splits.py`, positive cases from 2021–2023 go to training and positive cases from 2024 go to testing. Negative cases are shuffled with seed 42; 118 are assigned to testing and the remainder to training. By default, the script verifies the supplied splits. It writes splits only when `--write` is explicitly specified. Because directory iteration order may affect negative sampling, use the supplied split files to reproduce the released row order. Images, financial records, and features are matched by six-digit company codes.

## 2. RF features

```bash
python train_rf.py
```

RF uses `data/standardized_financial_sentiment_features.csv` and fits only the 504 training companies listed in `splits/train.txt`. Its 24 inputs begin at column 6: `pos` is the sentiment score computed with SKEP from the MD&A text of annual reports, and the other 23 columns are financial indicators. The table contains the computed sentiment scores. `rf_features.py` fixes and checks the feature names and order. The RF has 70 trees, maximum depth 4, minimum leaf size 5, no class weighting, and seed 42. A One-Hot encoder is fitted to the training companies' leaf indices. The fitted artifacts are `saved_models/rf/rf_model.joblib` and `saved_models/rf/ohe_encoder.joblib`; test companies do not participate in fitting. To refit, run `python train_rf.py --force`.

## 3. T2T-ViT: choosing the training epoch by five-fold validation

`t2t-vit/` contains the image encoder, Transformer block, and image loading utilities in `visual_data.py`. `select_vit_epochs.py` chooses an encoder training epoch using validation folds drawn only from the training split:

1. Split the 504 training companies with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Fold `k` uses model seed `42 + k`. Reinitialize the T2T-ViT encoder and a `256 -> 2` fully connected classification layer for each fold.
2. Train each fold for 100 epochs. The encoder produces 256-dimensional features and has four Transformer blocks. Resize images to `224 x 224`. Training augmentation applies up to 5% horizontal translation and a scale factor of 0.95–1.05. Validation and feature extraction only resize and convert images to tensors. No horizontal flip is used.
3. Use cross-entropy loss and AdamW with learning rate `3e-4`, weight decay `1e-4`, and batch size 32. The cosine learning-rate scheduler has `T_max=100`. Record validation AUC after each epoch in each fold.
4. For each epoch, average the validation AUC across the five folds. Choose the epoch with the highest mean AUC, using the earliest epoch to break ties. This does not average the five folds' individually best epochs or use test labels to choose an epoch.

```text
mean_auc(e) = (auc_fold0(e) + ... + auc_fold4(e)) / 5
selected_epoch = argmax_{e=1,...,100} mean_auc(e)
```

The commands below write fold histories and the mean AUC curve to `runs/vit_cv/`, then write the encoder and vectors used by later stages to `saved_models/vit/`:

```bash
for fold in 0 1 2 3 4; do
  python select_vit_epochs.py --fold "$fold" --epochs 100 \
    --output_json "runs/vit_cv/fold${fold}.json"
done
python select_vit_epochs.py --select_cv runs/vit_cv --epochs 100
SELECTED_EPOCH=$(python -c "import json; print(json.load(open('runs/vit_cv/selection.json'))['selected_epoch'])")
python select_vit_epochs.py --selected_epoch "$SELECTED_EPOCH" \
  --epochs 100 --save_dir saved_models/vit \
  --output_json runs/vit_cv/final_metrics.json
```

The released selection chose epoch **71**. A freshly initialized model is then trained on all 504 training companies for 71 epochs while keeping the scheduler period at 100 epochs. `saved_models/vit/` contains `full_encoder.pth`, `train_vectors.npy`, and `test_vectors.npy`. The `.npy` files contain 256-dimensional image features in the exact line order of `splits/train.txt` and `splits/test.txt`; the arrays do not contain company codes. The fusion script checks the fixed split files' SHA-256 digests so that a changed list or row order cannot silently misalign features. Rerunning this stage overwrites files with the same names.

## 4. RF and T2T-ViT fusion

```bash
python train_rf_vit_classifier.py
```

The script verifies the fixed split files, matches financial rows by six-digit company code, and reads ViT vectors in split-file order. It concatenates **T2T-ViT's 256 dimensions first, followed by RF's 908 dimensions** to produce 1164-dimensional vectors. `train_fused_vectors.npy` and `test_fused_vectors.npy` follow the exact line order of `splits/train.txt` and `splits/test.txt`. The auxiliary classifier is `1164 -> 128 -> 32 -> 2`, with ReLU and dropout 0.60 after each hidden layer. It uses AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 32, seed 42, and six training epochs. This classifier is used to evaluate RF+ViT without the graph; HGT reads the fused vectors, not this classifier's weights. After running the ViT stage above, `python train_rf_vit_classifier.py --vit_metrics_json runs/vit_cv/final_metrics.json` also writes standalone ViT metrics into the newly generated `results.json`.

`saved_models/rf_vit_classifier/` contains `train_fused_vectors.npy`, `test_fused_vectors.npy`, `classifier.pth`, `test_probabilities.npy`, and `results.json`. HGT reads company codes directly from the fixed `splits/` files and verifies their SHA-256 digests; no additional code-list text files are needed for these vectors.

## 5. Semantic vectors for non-company nodes

```bash
python rebuild_text_cache.py
```

`semantic_node_texts.py` builds Chinese descriptions of concept, executive, shareholder, and subsidiary nodes from the relationship tables. `rebuild_text_cache.py` encodes them with `BAAI/bge-small-zh-v1.5` into four sets of 512-dimensional vectors in `cache/*_embeddings.npy`. Each matching `.npy.json` manifest records the model name, node count, dimension, and SHA-256 digest of the ordered texts; the graph loader verifies these manifests. The vectors and manifests are included. Regeneration requires access to the BGE pretrained model.

## 6. HGT training and evaluation

```bash
python run_hgt_saved_features.py --epochs 100 \
  --lr-decay-epochs 100 --heads 2 --classifier-kind linear \
  --out-dir saved_models/hgt_reproduced
```

`graph_data_hetero.py` joins each company's 1164-dimensional fused vector to its graph node by company code and loads the 512-dimensional semantic vectors for the four other node types. At the HGT input, company features are ordered **RF's 908 dimensions first, then T2T-ViT's 256 dimensions**. Separate two-layer projections map company and non-company inputs to 128 dimensions. `hgt_model.py` uses two HGT layers, two attention heads per layer, and an independently sampled 10% edge dropout mask in each layer. A learnable residual coefficient combines the graph branch with the initial company projection; the initial graph-branch weight is 0.45. The final classifier is a single `128 -> 2` linear layer.

Training uses AdamW with initial learning rate `3e-4`, weight decay `1e-4`, a 100-epoch cosine decay to a minimum learning rate of `1e-6`, batch size 32, seed 42, feature dropout 0.60, gradient-norm clipping at 1, and ordinary cross-entropy (zero label smoothing). The binary decision threshold is 0.3. Only training labels are used for gradient updates.

The command saves the final-epoch model to `saved_models/hgt_reproduced/final_model.pth` and writes parameters and final-epoch training and test metrics to `results.json`. Add `--record-epoch-results` to record metrics for every epoch, or `--save-epoch-checkpoints` to save every epoch's weights.

## 7. File map

| Path | Contents |
| --- | --- |
| `splits/` | Fixed lists of training and test companies and labels |
| `saved_models/rf/` | RF and leaf-node One-Hot encoder |
| `saved_models/vit/` | Encoder weights trained on the full training split and two 256-dimensional vector arrays |
| `saved_models/rf_vit_classifier/` | Fused 1164-dimensional vectors and RF+ViT classifier |
| `cache/` | BGE vectors and validation manifests for four non-company node types |
