"""Regenerate the four non-company node caches with real BGE embeddings."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from graph_data_hetero import _read_csv, text_digest
from semantic_node_texts import build_node_texts, company_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", default="data/concepts.csv")
    parser.add_argument("--exec", dest="executive", default="data/executives.csv")
    parser.add_argument("--holder", default="data/shareholders.csv")
    parser.add_argument("--subs", default="data/subsidiaries.csv")
    parser.add_argument("--finance", default="data/standardized_financial_sentiment_features.csv")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    descriptions = build_node_texts(
        _read_csv(args.concept), _read_csv(args.executive),
        _read_csv(args.holder), _read_csv(args.subs), company_names(args.finance)
    )
    model_name = "BAAI/bge-small-zh-v1.5"
    model = SentenceTransformer(model_name)
    os.makedirs(args.cache_dir, exist_ok=True)
    backup_dir = os.path.join(args.cache_dir, "previous_embeddings_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(backup_dir, exist_ok=True)

    for node_type in ("concept", "executive", "shareholder", "subsidiary"):
        texts = descriptions[node_type]
        print(f"Encoding {node_type}: {len(texts)} nodes", flush=True)
        vectors = np.asarray(model.encode(texts, batch_size=args.batch_size,
                                          show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
        if vectors.shape != (len(texts), 512) or not np.isfinite(vectors).all():
            raise ValueError(f"Unexpected {node_type} embeddings: {vectors.shape}")
        path = os.path.join(args.cache_dir, f"{node_type}_embeddings.npy")
        manifest_path = path + ".json"
        for old in (path, manifest_path):
            if os.path.exists(old):
                shutil.copy2(old, backup_dir)
        temporary_path = path + ".tmp.npy"
        np.save(temporary_path, vectors)
        os.replace(temporary_path, path)
        manifest = {"model": model_name, "node_type": node_type,
                    "count": len(texts), "dimension": 512,
                    "text_sha256": text_digest(texts)}
        temporary_manifest = manifest_path + ".tmp"
        with open(temporary_manifest, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        os.replace(temporary_manifest, manifest_path)
        print(f"Saved {path} {vectors.shape}", flush=True)
    print(f"Previous caches saved in {backup_dir}", flush=True)


if __name__ == "__main__":
    main()
