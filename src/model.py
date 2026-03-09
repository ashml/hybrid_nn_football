from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def time_based_split(dataset: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> SplitData:
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    feature_cols = [c for c in dataset.columns if c not in {"target"}]

    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]

    return SplitData(
        x_train=train[feature_cols],
        y_train=train["target"],
        x_val=val[feature_cols],
        y_val=val["target"],
        x_test=test[feature_cols],
        y_test=test["target"],
    )


class HybridNeuralNetworkModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        batch_size=32,
                        max_iter=100,
                        random_state=42,
                    ),
                ),
            ]
        )

    def train_model(self, split_data: SplitData) -> Dict[str, float]:
        self.pipeline.fit(split_data.x_train, split_data.y_train)

        val_pred = self.pipeline.predict(split_data.x_val)
        val_proba = self.pipeline.predict_proba(split_data.x_val)

        metrics = {
            "val_accuracy": accuracy_score(split_data.y_val, val_pred),
            "val_f1_macro": f1_score(split_data.y_val, val_pred, average="macro"),
            "val_log_loss": log_loss(split_data.y_val, val_proba, labels=[0, 1, 2]),
        }
        return metrics

    def evaluate(self, x: pd.DataFrame, y: pd.Series, prefix: str = "test") -> Dict[str, float | list]:
        pred = self.pipeline.predict(x)
        proba = self.pipeline.predict_proba(x)

        return {
            f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_f1_macro": f1_score(y, pred, average="macro"),
            f"{prefix}_log_loss": log_loss(y, proba, labels=[0, 1, 2]),
            f"{prefix}_confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(features)
