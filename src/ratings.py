from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class EloRatingSystem:
    """Dynamic team-rating system based on Elo with home advantage."""

    base_rating: float = 1000.0
    k_factor: float = 20.0
    home_advantage: float = 100.0
    ratings: Dict[str, float] = field(default_factory=dict)

    def initialize_team(self, team: str) -> None:
        if team not in self.ratings:
            self.ratings[team] = self.base_rating

    def get_rating(self, team: str) -> float:
        self.initialize_team(team)
        return self.ratings[team]

    def expected_score(self, home_team: str, away_team: str) -> Tuple[float, float]:
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        exp_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
        exp_away = 1.0 - exp_home
        return exp_home, exp_away

    @staticmethod
    def score_from_goals(home_goals: int, away_goals: int) -> Tuple[float, float]:
        if home_goals > away_goals:
            return 1.0, 0.0
        if home_goals < away_goals:
            return 0.0, 1.0
        return 0.5, 0.5

    def update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """Update ratings after a match and return new ratings."""
        self.initialize_team(home_team)
        self.initialize_team(away_team)

        exp_home, exp_away = self.expected_score(home_team, away_team)
        score_home, score_away = self.score_from_goals(home_goals, away_goals)

        self.ratings[home_team] += self.k_factor * (score_home - exp_home)
        self.ratings[away_team] += self.k_factor * (score_away - exp_away)

        return self.ratings[home_team], self.ratings[away_team]
