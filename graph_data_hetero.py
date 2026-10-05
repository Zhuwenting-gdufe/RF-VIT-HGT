# graph_data_hetero.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from semantic_node_texts import build_node_texts, company_names


def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "gb18030", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _uniq_keep_order(xs: List[str]) -> List[str]:
    seen = set();
    out = []
    for x in xs:
        if x not in seen:
            out.append(x);
            seen.add(x)
    return out


def _compact(s: str, max_len=64) -> str:
    s = str(s).replace("\n", " ").strip()
    return s[:max_len]


def text_digest(texts: List[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_text_cache(texts: List[str], cache_path: str) -> np.ndarray:
    """Require the BGE cache to match the exact descriptions and node order."""
    manifest_path = cache_path + ".json"
    if not os.path.exists(cache_path) or not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing BGE cache {cache_path}; run python rebuild_text_cache.py")
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    if (manifest.get("model") != "BAAI/bge-small-zh-v1.5"
            or manifest.get("text_sha256") != text_digest(texts)):
        raise ValueError(f"Stale or incompatible BGE cache {cache_path}; run python rebuild_text_cache.py")
    vectors = np.load(cache_path, allow_pickle=False)
    if vectors.shape != (len(texts), 512) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid BGE cache shape/content: {cache_path}: {vectors.shape}")
    print(f"Loading verified BGE cache: {cache_path} {vectors.shape}")
    return vectors


def apply_edge_dropout(hetero_data, drop_rate=0.1):
    """Apply edge-drop augmentation"""
    if drop_rate <= 0:
        return hetero_data

    augmented_data = hetero_data.clone()
    dropped_edges_count = 0
    total_edges_count = 0

    for edge_type in hetero_data.edge_index_dict.keys():
        edge_index = hetero_data.edge_index_dict[edge_type]
        if edge_index.size(1) > 0:
            total_edges_count += edge_index.size(1)
            # Sample edges to retain.
            keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
            kept_edges = edge_index[:, keep_mask]
            dropped_edges_count += (edge_index.size(1) - kept_edges.size(1))

            augmented_data[edge_type].edge_index = kept_edges

    if dropped_edges_count > 0:
        print(f"Graph augmentation: removed {dropped_edges_count}/{total_edges_count} edges (drop rate: {drop_rate:.1%})")

    return augmented_data


def build_hetero_graph_with_augmentation(
        concept_csv: str,
        exec_csv: str,
        holder_csv: str,
        subs_csv: str,
        rf_code2vec: Dict[str, np.ndarray],
        kline_code2vec: Dict[str, np.ndarray],
        code2label_txt: Optional[Dict[str, int]] = None,
        use_edges: Dict[str, bool] = None,
        edge_drop_rate: float = 0.1,
        training: bool = False,  # training-mode flag
        finance_csv: str = "data/standardized_financial_sentiment_features.csv",
):
    """Build a heterogeneous graph with optional edge dropout"""

    # Build the base graph.
    hetero, meta = build_hetero_graph(
        concept_csv, exec_csv, holder_csv, subs_csv,
        rf_code2vec, kline_code2vec, code2label_txt, use_edges, finance_csv
    )

    # Apply edge dropout only during training.
    if training and edge_drop_rate > 0:
        hetero = apply_edge_dropout(hetero, edge_drop_rate)

    return hetero, meta


def build_hetero_graph(
        concept_csv: str,
        exec_csv: str,
        holder_csv: str,
        subs_csv: str,
        rf_code2vec: Dict[str, np.ndarray],
        kline_code2vec: Dict[str, np.ndarray],
        code2label_txt: Optional[Dict[str, int]] = None,
        use_edges: Dict[str, bool] = None,
        finance_csv: str = "data/standardized_financial_sentiment_features.csv",
) -> Tuple[HeteroData, dict]:
    if use_edges is None:
        # Default relation switches.
        use_edges = {
            "concept": True,
            "executive": True,
            "shareholder": True,
            "subsidiary": True,
        }
    df_cpt = _read_csv(concept_csv)
    df_exe = _read_csv(exec_csv)
    df_hld = _read_csv(holder_csv)
    df_sub = _read_csv(subs_csv)

    for df in (df_cpt, df_exe, df_hld, df_sub):
        first_column = df.columns[0]
        df[first_column] = df[first_column].astype(str).str.strip()

    def normalize_code(code: str) -> str:
        code = str(code).strip()
        if code.isdigit():
            code = code.zfill(6)
        return code

    normalized_rf = {normalize_code(k): v for k, v in rf_code2vec.items()}
    normalized_kline = {normalize_code(k): v for k, v in kline_code2vec.items()}
    rf_set = set(normalized_rf)
    kline_set = set(normalized_kline)
    cpt_set = set(normalize_code(k) for k in df_cpt.iloc[:, 0])
    exec_set = set(normalize_code(k) for k in df_exe.iloc[:, 0])
    holder_set = set(normalize_code(k) for k in df_hld.iloc[:, 0])
    companies = sorted(rf_set | kline_set | cpt_set | exec_set | holder_set)

    concepts = _uniq_keep_order(df_cpt.iloc[:, 1].astype(str).tolist())
    exec_names = _uniq_keep_order(df_exe.iloc[:, 1].astype(str).tolist())
    holders = _uniq_keep_order(df_hld.iloc[:, 1].astype(str).tolist())
    subs = _uniq_keep_order(df_sub.iloc[:, 1].astype(str).tolist())

    lid = {
        "company": {c: i for i, c in enumerate(companies)},
        "concept": {c: i for i, c in enumerate(concepts)},
        "executive": {c: i for i, c in enumerate(exec_names)},
        "shareholder": {c: i for i, c in enumerate(holders)},
        "subsidiary": {c: i for i, c in enumerate(subs)},
    }

    rf_dim = len(next(iter(rf_code2vec.values()))) if rf_code2vec else 0
    k_dim = len(next(iter(kline_code2vec.values()))) if kline_code2vec else 0
    comp_dim = rf_dim + k_dim

    cache_dir = "cache"
    descriptions = build_node_texts(df_cpt, df_exe, df_hld, df_sub, company_names(finance_csv))
    assert len(descriptions["concept"]) == len(concepts)
    assert len(descriptions["executive"]) == len(exec_names)
    assert len(descriptions["shareholder"]) == len(holders)
    assert len(descriptions["subsidiary"]) == len(subs)
    vec_concept = load_text_cache(descriptions["concept"], f"{cache_dir}/concept_embeddings.npy") if concepts else []
    llm_dim = vec_concept.shape[1] if len(vec_concept) > 0 else 512
    vec_exec = load_text_cache(descriptions["executive"], f"{cache_dir}/executive_embeddings.npy") if exec_names else []
    vec_holder = load_text_cache(descriptions["shareholder"], f"{cache_dir}/shareholder_embeddings.npy") if holders else []
    vec_subs = load_text_cache(descriptions["subsidiary"], f"{cache_dir}/subsidiary_embeddings.npy") if subs else []

    data = HeteroData()
    comp_x = np.zeros((len(companies), comp_dim), dtype="float32") if comp_dim > 0 else np.zeros((len(companies), 1),
                                                                                                 dtype="float32")
    for c, i in lid["company"].items():
        r = normalized_rf.get(c, np.zeros((rf_dim,), dtype="float32")) if rf_dim > 0 else np.zeros((0,), dtype="float32")
        k = normalized_kline.get(c, np.zeros((k_dim,), dtype="float32")) if k_dim > 0 else np.zeros((0,), dtype="float32")
        if comp_dim > 0:
            comp_x[i] = np.concatenate([r, k], axis=-1) if (k_dim > 0 or rf_dim > 0) else np.zeros((comp_dim,),
                                                                                                   dtype="float32")
    data["company"].x = torch.tensor(comp_x, dtype=torch.float32)

    data["concept"].x = torch.tensor(vec_concept, dtype=torch.float32) if concepts else torch.zeros((0, llm_dim),
                                                                                                    dtype=torch.float32)
    data["executive"].x = torch.tensor(vec_exec, dtype=torch.float32) if exec_names else torch.zeros((0, llm_dim),
                                                                                                     dtype=torch.float32)
    data["shareholder"].x = torch.tensor(vec_holder, dtype=torch.float32) if holders else torch.zeros((0, llm_dim),
                                                                                                      dtype=torch.float32)
    data["subsidiary"].x = torch.tensor(vec_subs, dtype=torch.float32) if subs else torch.zeros((0, llm_dim),
                                                                                                dtype=torch.float32)

    # -------------------- Bidirectional relation construction --------------------
    def add_edges(df, src_t, dst_t, edge_name, rev_edge_name, enabled=True):
        if not enabled:
            data[(src_t, edge_name, dst_t)].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[(dst_t, rev_edge_name, src_t)].edge_index = torch.zeros((2, 0), dtype=torch.long)
            print(f"Disabled relation: ({src_t},{edge_name},{dst_t}) / ({dst_t},{rev_edge_name},{src_t})")
            return
        if df is None or df.empty:
            data[(src_t, edge_name, dst_t)].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[(dst_t, rev_edge_name, src_t)].edge_index = torch.zeros((2, 0), dtype=torch.long)
            return
        src_idx, dst_idx = [], []
        for _, r in df.iterrows():
            src_key = normalize_code(r.iloc[0]) if src_t == "company" else str(r.iloc[0]).strip()
            dst_key = normalize_code(r.iloc[1]) if dst_t == "company" else str(r.iloc[1]).strip()
            if src_key in lid[src_t] and dst_key in lid[dst_t]:
                src_idx.append(lid[src_t][src_key])
                dst_idx.append(lid[dst_t][dst_key])
        e = torch.tensor([src_idx, dst_idx], dtype=torch.long) if src_idx else torch.zeros((2, 0), dtype=torch.long)
        rev_e = torch.tensor([dst_idx, src_idx], dtype=torch.long) if src_idx else torch.zeros((2, 0), dtype=torch.long)
        data[(src_t, edge_name, dst_t)].edge_index = e
        data[(dst_t, rev_edge_name, src_t)].edge_index = rev_e
        print(f"Built relation: ({src_t},{edge_name},{dst_t}) | count: {e.size(1)}")
        print(f"Built reverse relation: ({dst_t},{rev_edge_name},{src_t}) | count: {rev_e.size(1)}")

    add_edges(df_cpt, "company", "concept", "has_concept", "rev_has_concept", enabled=use_edges.get("concept", True))
    add_edges(df_exe, "company", "executive", "has_executive", "rev_has_executive",
              enabled=use_edges.get("executive", True))
    add_edges(df_hld, "company", "shareholder", "has_shareholder", "rev_has_shareholder",
              enabled=use_edges.get("shareholder", True))
    add_edges(df_sub, "company", "subsidiary", "has_subsidiary", "rev_has_subsidiary",
              enabled=use_edges.get("subsidiary", True))

    # -------------------- Company labels --------------------
    y = torch.tensor([code2label_txt.get(c, 0) for c in companies], dtype=torch.long)
    data["company"].y = y

    meta = {"lid": lid}

    return data, meta
