from __future__ import annotations

import argparse

from system import FootballForecastSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict football match outcome")
    parser.add_argument("--model", type=str, default="model.pkl", help="Path to trained model")
    parser.add_argument("--home-team", type=str, required=True)
    parser.add_argument("--away-team", type=str, required=True)
    args = parser.parse_args()

    system = FootballForecastSystem.load(args.model)
    prediction = system.predict(args.home_team, args.away_team)
    print(prediction)


if __name__ == "__main__":
    main()
