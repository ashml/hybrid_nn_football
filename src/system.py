from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from features import FeatureBuilder, build_feature_dataset
from model import HybridNeuralNetworkModel, time_based_split
from ratings import EloRatingSystem


REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "tournament",
]


@dataclass
class FootballForecastSystem:
    rating_system: EloRatingSystem
    feature_builder: FeatureBuilder
    model: HybridNeuralNetworkModel
    feature_columns: list[str] | None = None

    @classmethod
    def create(cls) -> "FootballForecastSystem":
        rating_system = EloRatingSystem()
        return cls(
            rating_system=rating_system,
            feature_builder=FeatureBuilder(rating_system),
            model=HybridNeuralNetworkModel(),
        )

    def load_data(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date", ascending=True).reset_index(drop=True)

    def train_model(self, data: pd.DataFrame) -> Dict[str, float | list]:
        dataset = build_feature_dataset(data)
        self.feature_columns = [c for c in dataset.columns if c != "target"]
        split = time_based_split(dataset)

        metrics = self.model.train_model(split)
        metrics.update(self.model.evaluate(split.x_test, split.y_test, prefix="test"))

        # Rebuild full rating/state for future inference context.
        self.rating_system = EloRatingSystem()
        self.feature_builder = FeatureBuilder(self.rating_system)
        for _, match in data.iterrows():
            self.feature_builder.build_features(match)
            self.feature_builder.update_after_match(match)

        return metrics

    def update_ratings(self, match: Dict) -> None:
        series = pd.Series(match)
        self.feature_builder.update_after_match(series)

    def build_features(self, match: Dict) -> Dict[str, float]:
        series = pd.Series(match)
        return self.feature_builder.build_features(series)

    def predict(self, home_team: str, away_team: str) -> Dict[str, float]:
        if self.feature_columns is None:
            raise RuntimeError("Model is not trained yet.")

        match = {
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": 0,
            "away_goals": 0,
        }
        feature_row = self.build_features(match)
        feature_df = pd.DataFrame([feature_row])[self.feature_columns]

        proba = self.model.predict(feature_df)[0]
        return {
            "home_win": float(proba[2]),
            "draw": float(proba[1]),
            "away_win": float(proba[0]),
        }

    def save(self, path: str | Path) -> None:
        obj = {
            "rating_system": self.rating_system,
            "feature_builder": self.feature_builder,
            "model_pipeline": self.model.pipeline,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(obj, path)

    @staticmethod
    def load(path: str | Path) -> "FootballForecastSystem":
        obj = joblib.load(path)
        system = FootballForecastSystem(
            rating_system=obj["rating_system"],
            feature_builder=obj["feature_builder"],
            model=HybridNeuralNetworkModel(),
            feature_columns=obj["feature_columns"],
        )
        system.model.pipeline = obj["model_pipeline"]
        return system

    @staticmethod
    def save_metrics(metrics: Dict, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
