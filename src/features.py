from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np
import pandas as pd

from ratings import EloRatingSystem


@dataclass
class TeamStatsState:
    goals_for: int = 0
    goals_against: int = 0
    matches: int = 0
    recent_points: Deque[int] = field(default_factory=lambda: deque(maxlen=5))


class FeatureBuilder:
    """Builds leakage-safe pre-match features and updates state after each match."""

    def __init__(self, rating_system: EloRatingSystem):
        self.rating_system = rating_system
        self.team_stats: Dict[str, TeamStatsState] = defaultdict(TeamStatsState)

    def _avg_goals_for(self, team: str) -> float:
        st = self.team_stats[team]
        return st.goals_for / st.matches if st.matches > 0 else 0.0

    def _avg_goals_against(self, team: str) -> float:
        st = self.team_stats[team]
        return st.goals_against / st.matches if st.matches > 0 else 0.0

    def _last5_points(self, team: str) -> float:
        st = self.team_stats[team]
        if not st.recent_points:
            return 0.0
        return float(sum(st.recent_points))

    def build_features(self, match: pd.Series) -> Dict[str, float]:
        home_team = match["home_team"]
        away_team = match["away_team"]

        rating_home = self.rating_system.get_rating(home_team)
        rating_away = self.rating_system.get_rating(away_team)

        feature_row = {
            "rating_home": rating_home,
            "rating_away": rating_away,
            "rating_difference": rating_home - rating_away,
            "average_goals_scored_home": self._avg_goals_for(home_team),
            "average_goals_conceded_home": self._avg_goals_against(home_team),
            "average_goals_scored_away": self._avg_goals_for(away_team),
            "average_goals_conceded_away": self._avg_goals_against(away_team),
            "last_5_matches_points_home": self._last5_points(home_team),
            "last_5_matches_points_away": self._last5_points(away_team),
        }

        return feature_row

    def update_after_match(self, match: pd.Series) -> None:
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        self.rating_system.update_ratings(home_team, away_team, home_goals, away_goals)

        home_state = self.team_stats[home_team]
        away_state = self.team_stats[away_team]

        home_state.goals_for += home_goals
        home_state.goals_against += away_goals
        home_state.matches += 1

        away_state.goals_for += away_goals
        away_state.goals_against += home_goals
        away_state.matches += 1

        if home_goals > away_goals:
            home_points, away_points = 3, 0
        elif home_goals < away_goals:
            home_points, away_points = 0, 3
        else:
            home_points, away_points = 1, 1

        home_state.recent_points.append(home_points)
        away_state.recent_points.append(away_points)


def match_result(home_goals: int, away_goals: int) -> int:
    if home_goals < away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def build_feature_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    rating_system = EloRatingSystem()
    builder = FeatureBuilder(rating_system)

    rows: List[Dict[str, float]] = []
    targets: List[int] = []

    for _, match in matches.iterrows():
        rows.append(builder.build_features(match))
        targets.append(match_result(int(match["home_goals"]), int(match["away_goals"])))
        builder.update_after_match(match)

    features_df = pd.DataFrame(rows)
    features_df["target"] = np.array(targets)
    return features_df
