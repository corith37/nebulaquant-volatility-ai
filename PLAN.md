# NebulaQuant Volatility AI — Project Plan

## 1. Project Overview

NebulaQuant Volatility AI is a machine learning quant research application designed to identify high-volatility stock setups and predict their likely future outcomes.

The first version uses historical stock data, technical indicators, supervised machine learning, and backtesting to evaluate whether volatility breakout signals have predictive value.

This project is for research, education, and paper trading. It should not be used for real-money trading until thoroughly validated.

---

## 2. Core Objective

The goal is to build a model that scans a universe of stocks and predicts one of three outcomes over the next five trading days:

```text
0 = No major move / sideways
1 = Upside volatility breakout
2 = Downside volatility breakdown
```

The app then backtests these predictions and displays the best current setups in a dashboard.

---

## 3. Target Prediction

For each stock and date, the model predicts the future 5-day return category.

### Label Logic

```python
future_return_5d = close.shift(-5) / close - 1

if future_return_5d > 0.04:
    label = 1
elif future_return_5d < -0.04:
    label = 2
else:
    label = 0
```

The 4% threshold is configurable in `config.yaml`.

---

## 4. Technology Stack

- Python 3.11+
- pandas, numpy
- yfinance
- scikit-learn
- xgboost
- ta (technical indicators)
- plotly
- streamlit
- joblib
- pytest

Future optional upgrades:

- PyTorch LSTM / Transformer
- Alpaca paper trading
- Polygon.io market data

---

## 5. Folder Structure

```text
nebulaquant-volatility-ai/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/{raw,processed,predictions}/
├── models/saved/
├── reports/{metrics,backtests,charts}/
├── src/
├── app/
├── scripts/
└── tests/
```

---

## 6. Configuration

`config.yaml` holds project, data, labels, model, backtest, and scanner parameters.

---

## 7. Data Pipeline

1. Download OHLCV with yfinance -> `data/raw/{ticker}.csv`
2. Calculate features -> `data/processed/{ticker}_features.csv`
3. Create labels (3-class future return).
4. Combine all into `data/processed/model_dataset.csv`.

---

## 8. Feature List

- Volatility: ATR, ATR%, daily range %, 5/10/20-day rolling vol, Bollinger Band width, gap %
- Volume: 20d avg volume, relative volume, volume spike ratio, dollar volume
- Momentum: RSI, MACD/signal/hist, 5/10/20-day returns, ROC
- Trend: SMA 20/50/200, price-above flags, SMA slope, distance from 20d/52w high/low
- Market context (SPY): 5/20-day return, rolling vol, above SMA 50

---

## 9. Model Training

- XGBoost classifier (3 classes).
- Chronological split: 70% train / 15% val / 15% test (no shuffle).
- Save model to `models/saved/xgboost_volatility_model.joblib`.
- Save feature columns to `models/saved/feature_columns.json`.

---

## 10. Evaluation Metrics

- accuracy, precision, recall, F1
- confusion matrix
- classification report
- feature importance

Saved to `reports/metrics/`.

---

## 11. Backtesting System

### Long signal
```text
predicted_class == 1
upside_breakout_probability >= probability_threshold
relative_volume >= 1.2
close > SMA 20
```

### Exit
take profit / stop loss / max hold days reached.

### Defaults
$10,000 capital, 10% position, +6% TP, -3% SL, 5d max hold, 0.1% commission, 0.1% slippage.

### Outputs
- `reports/backtests/trade_log.csv`
- `reports/backtests/backtest_metrics.csv`

Metrics: total return, final equity, win rate, # trades, avg win/loss, profit factor, max drawdown, Sharpe, avg hold days, best/worst trade.

---

## 12. Scanner

1. Load latest data + saved model.
2. Compute latest features.
3. Predict class probabilities.
4. Rank by upside breakout probability.
5. Save `data/predictions/latest_predictions.csv`.

### Suggested action
- Strong Long Watch: upside >= 0.75, rel vol >= 1.5, close > SMA20
- Long Watch: upside >= 0.65, close > SMA20
- Downside Watch: downside >= 0.65, close < SMA20
- No Trade: otherwise

---

## 13. Streamlit Dashboard

Tabs: Overview, Scanner, Stock Detail, Backtest Results, Model Metrics, Settings.

Run with `streamlit run app/streamlit_app.py`.

---

## 14. Command-Line Workflow

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train.py
python scripts/run_backtest.py
python scripts/run_scanner.py
streamlit run app/streamlit_app.py
```

---

## 15. Testing Plan

Pytest tests for feature generation, label generation, and backtest trade logic.

---

## 16. Development Phases

1. Project setup
2. Data loader
3. Feature engineering
4. Label creation
5. Dataset builder
6. Model training
7. Model evaluation
8. Backtesting
9. Scanner
10. Streamlit dashboard
11. Tests
12. Documentation

---

## 17. Future Upgrade Roadmap

- Better data: Polygon.io, Alpaca, Nasdaq Data Link
- Better models: LightGBM, CatBoost, LSTM, Transformer, ensembles
- Better targets: 1/3/10-day, realized vol, breakout follow-through, fakeout
- Market regime: SPY/QQQ trend, VIX, sector strength, breadth
- Paper trading via Alpaca (with daily loss limit, max positions, no live trading)
- PyTorch sequence model: 60-bar window -> 3 classes

---

## 18. Risk Controls

- Predictions are never guaranteed.
- Always use stop loss.
- Always track drawdown.
- Always paper trade first.
- Avoid overfitting.

---

## 19. Definition of Done

Data downloads, features/labels generate, dataset builds, model trains and saves, evaluation reports save, backtest runs, trade log saves, scanner ranks stocks, dashboard opens, README explains usage, tests pass.

---

## 20. Main Execution Commands

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_dataset.py
python scripts/train.py
python scripts/run_backtest.py
python scripts/run_scanner.py
streamlit run app/streamlit_app.py
```
