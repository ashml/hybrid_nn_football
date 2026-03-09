from __future__ import annotations

import argparse
from pathlib import Path

from system import FootballForecastSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid NN football forecast model")
    parser.add_argument("--data", type=str, default="data/matches.csv", help="Path to matches CSV")
    parser.add_argument("--model-out", type=str, default="model.pkl", help="Output path for model file")
    parser.add_argument("--metrics-out", type=str, default="reports/metrics.json", help="Output metrics path")
    args = parser.parse_args()

    system = FootballForecastSystem.create()
    matches = system.load_data(args.data)
    metrics = system.train_model(matches)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    system.save(args.model_out)
    system.save_metrics(metrics, args.metrics_out)

    print("Training complete")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
