"""Chinese node descriptions following Table 4 and Algorithm 2 of the paper."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import pandas as pd


def normalize_code(value) -> str:
    value = str(value).strip()
    return value.zfill(6) if value.isdigit() else value


def company_names(finance_csv: str) -> Dict[str, str]:
    for encoding in ("utf-8", "gb18030", "utf-8-sig"):
        try:
            frame = pd.read_csv(finance_csv, encoding=encoding, dtype={"股票代码": str})
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode company names from {finance_csv}")
    required = {"股票代码", "公司全称"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{finance_csv} must contain columns {sorted(required)}")
    names = {}
    for code, name in zip(frame["股票代码"], frame["公司全称"]):
        if pd.notna(name) and str(name).strip():
            names[normalize_code(code)] = str(name).strip()
    return names


def _company(code, names: Dict[str, str]) -> str:
    key = normalize_code(code)
    if key not in names:
        raise ValueError(f"Missing company name for stock code {key}")
    return names[key]


def _rank(value) -> str:
    if pd.isna(value):
        raise ValueError("Missing shareholder rank")
    number = int(float(value))
    return f"第{number}大股东"


def _percentage(value) -> str:
    if pd.isna(value):
        raise ValueError("Missing shareholder percentage")
    return f"{float(str(value).replace('%', '').strip()):.2f}%"


def build_node_texts(df_cpt, df_exe, df_hld, df_sub, names: Dict[str, str]):
    """One text per unique node; append one full sentence per linked company."""
    result = {}

    concepts = df_cpt.iloc[:, 1].astype(str).drop_duplicates().tolist()
    result["concept"] = [f"{name}是一个行业。" for name in concepts]

    executive = defaultdict(list)
    for row in df_exe.itertuples(index=False, name=None):
        code, name, role = row[:3]
        if pd.isna(role) or not str(role).strip():
            raise ValueError(f"Missing position for executive {name} at {code}")
        firm = _company(code, names)
        executive[str(name)].append(f"{name}是{firm}的高级管理人员，在{firm}担任{str(role).strip()}。")
    result["executive"] = ["".join(executive[name]) for name in executive]

    shareholder = defaultdict(list)
    for row in df_hld.itertuples(index=False, name=None):
        code, name, rank, percentage = row[:4]
        firm = _company(code, names)
        shareholder[str(name)].append(
            f"{name}是{firm}的股东，持有{_percentage(percentage)}的股份，是该公司{_rank(rank)}。"
        )
    result["shareholder"] = ["".join(shareholder[name]) for name in shareholder]

    subsidiary = defaultdict(list)
    for row in df_sub.itertuples(index=False, name=None):
        code, name = row[:2]
        subsidiary[str(name)].append(f"{name}是{_company(code, names)}的子公司。")
    result["subsidiary"] = ["".join(subsidiary[name]) for name in subsidiary]
    return result
