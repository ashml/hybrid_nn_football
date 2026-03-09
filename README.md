# Hybrid NN Football Forecast System

Реализация production-пайплайна для прогнозирования исходов футбольных матчей на основе:

- динамического рейтинга команд (Elo-подобная система с бонусом домашнего поля),
- гибридного набора признаков,
- feed-forward нейросети (MLP: 16 -> 8 -> 3) для 3-классовой классификации.

## Структура

- `src/ratings.py` — рейтинговая система и обновление рейтингов.
- `src/features.py` — формирование leakage-safe признаков.
- `src/model.py` — time-based split, обучение, метрики.
- `src/system.py` — end-to-end API:
  - `train_model(data)`
  - `update_ratings(match)`
  - `build_features(match)`
  - `predict(home_team, away_team)`
- `src/train.py` — CLI обучения.
- `src/predict.py` — CLI инференса.

## Требуемые поля данных

Минимальные колонки CSV:

- `match_id`
- `date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `tournament`

## Запуск

```bash
python src/train.py --data data/matches.csv --model-out model.pkl --metrics-out reports/metrics.json
python src/predict.py --model model.pkl --home-team "Team A" --away-team "Team B"
```

## Что сохраняется

- `model.pkl` — обученная модель + scaler + состояние рейтингов и статистик.
- `reports/metrics.json` — accuracy, log_loss, f1_score, confusion_matrix (test), а также валидационные метрики.
