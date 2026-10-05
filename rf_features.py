"""Random-forest leaf one-hot features for company nodes."""
from __future__ import annotations

import os
import random
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


RF_FEATURE_COLUMNS = (
    "pos", "可持续增长率", "每股净资产增长率", "净资产收益率增长率", "净利润增长率",
    "营业收入增长率", "总资产增长率", "利润总额增长率", "基本每股收益增长率", "营业利润增长率",
    "营业收入现金含量", "全部现金回收率", "营运指数", "现金适合比率", "现金再投资比率",
    "现金满足投资比率", "应收账款周转率", "存货周转率", "营业周期", "应付账款周转率",
    "营运资金(资本)周转率", "流动资产周转率", "总资产周转率", "股东权益周转率",
)


def checked_rf_features(df: pd.DataFrame) -> List[str]:
    """Keep the fitted RF's 24 input columns in their original order."""
    columns = df.columns[5:].tolist()
    if columns != list(RF_FEATURE_COLUMNS):
        raise ValueError("Financial feature columns or their order differ from the saved RF")
    return columns


def read_csv(path: str) -> pd.DataFrame:
    for encoding in ("utf-8", "gb18030", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    return pd.read_csv(path)


def normalize_stock_code(code) -> str:
    """Normalize Chinese stock codes to six characters."""
    text = str(code).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


class RFFeatureExtractor:
    """Fit/load an RF and encode each tree leaf assignment as one-hot."""

    def __init__(self, n_estimators=70, max_depth=4, random_state=42,
                 model_path="saved_models/rf/rf_model.joblib",
                 ohe_path="saved_models/rf/ohe_encoder.joblib",
                 deterministic=False,
                 class_weight=None, min_samples_leaf=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.model_path = model_path
        self.ohe_path = ohe_path
        self.deterministic = deterministic
        self.selected_features: List[str] = []
        self.rf = None
        self.ohe = None
        self.rf_dim = 0
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    def prepare(self, finance_csv: str, train_codes: List[str], force_retrain=False) -> None:
        paths = (self.model_path, self.ohe_path)
        if not force_retrain and all(os.path.exists(path) for path in paths):
            saved_rf = joblib.load(self.model_path)
            expected = {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
                "min_samples_leaf": self.min_samples_leaf,
            }
            features = checked_rf_features(self._preprocess(read_csv(finance_csv)))
            if (all(saved_rf.get_params()[key] == value for key, value in expected.items())
                    and saved_rf.n_features_in_ == len(features)):
                self.rf = saved_rf
                self.ohe = joblib.load(self.ohe_path)
                self.selected_features = features
                self.rf_dim = sum(len(category) for category in self.ohe.categories_)
                print(f"Loaded RF leaf encoder ({self.rf_dim} dimensions).")
                return
            print("Saved RF settings differ from the requested settings; refitting RF and leaf encoder.")
        self._train_and_save(finance_csv, train_codes)

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        # The dataset schema is positional so the original Chinese headers remain unchanged.
        df = df.rename(columns={df.columns[0]: "label", df.columns[2]: "company_name",
                                df.columns[3]: "stock_code"})
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        return df.fillna(0.0)

    def _train_and_save(self, finance_csv: str, train_codes: List[str]) -> None:
        if self.deterministic:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        df = self._preprocess(read_csv(finance_csv))
        train_set = {normalize_stock_code(code) for code in train_codes}
        train_df = df[df["stock_code"].isin(train_set)].copy()
        if train_df.empty:
            raise ValueError("No financial rows match the training stock codes.")
        self.selected_features = checked_rf_features(df)
        x_train = train_df[self.selected_features].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy()
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, class_weight=self.class_weight,
            min_samples_leaf=self.min_samples_leaf, n_jobs=1,
        ).fit(x_train, y_train)
        leaves = self.rf.apply(x_train)
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.ohe.fit(leaves)
        self.rf_dim = self.ohe.transform(leaves).shape[1]
        joblib.dump(self.rf, self.model_path)
        joblib.dump(self.ohe, self.ohe_path)
        print(f"Trained and saved RF leaf encoder ({self.rf_dim} dimensions).")

    def extract_features(self, codes: List[str], finance_csv: str) -> Dict[str, np.ndarray]:
        if self.rf is None or self.ohe is None:
            raise RuntimeError("Call prepare() before extract_features().")
        df = self._preprocess(read_csv(finance_csv))
        normalized = [normalize_stock_code(code) for code in codes]
        selected = df[df["stock_code"].isin(normalized)].copy()
        output = {code: np.zeros(self.rf_dim, dtype=np.float32) for code in normalized}
        if not selected.empty:
            x = selected[self.selected_features].to_numpy(dtype=np.float32)
            encoded = self.ohe.transform(self.rf.apply(x)).astype(np.float32)
            for code, vector in zip(selected["stock_code"], encoded):
                output[code] = vector
        found = sum(np.any(vector) for vector in output.values())
        print(f"Extracted RF features for {found}/{len(output)} companies.")
        return output
