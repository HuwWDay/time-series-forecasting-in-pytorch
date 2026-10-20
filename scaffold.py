"""
Time-Series Forecasting in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Time-series forecasting in PyTorch (Hands-On ML, chapter 13).

Story: forecast Chicago's daily rail ridership. Clean twenty years of data, set the
naive baselines, train a linear model and three recurrent ones on the same
windows, forecast two weeks ahead recursively and with a seq2seq LSTM, add bus
counts and day types as inputs, then save a forecaster and print next week's riders.
"""
import os
import tempfile
import numpy as np
import torch


def main() -> None:
    data = load_ridership()
    print(f"{len(data['dates'])} days, {data['dates'][0]} to {data['dates'][-1]}")
    _, train = select_range(data, "2016-01-01", "2018-12-31")
    val_dates, val = select_range(data, "2019-01-01", "2019-05-31")
    print(f"train {len(train)} days, validation {len(val)} days")
    print(f"naive baselines on validation: yesterday {naive_mae(val, 1) * 1e6:,.0f} riders, last week {naive_mae(val, 7) * 1e6:,.0f} riders")

    # ---- one-step models on the same windows ----
    results = compare_forecasters(train, val, window=56, epochs=20)
    for name in ["linear", "rnn", "gru", "lstm"]:
        print(f"  {name:7s} validation MAE {results[name] * 1e6:,.0f} riders")

    # ---- two weeks ahead ----
    X, Y = make_seq2seq_windows(train, 56, 14)
    Xv, Yv = make_seq2seq_windows(np.concatenate([train[-56:], val]), 56, 14)
    torch.manual_seed(0)
    s2s = Seq2SeqForecaster(hidden=32, horizon=14)
    per_h = train_seq2seq(s2s, X, Y, Xv, Yv, epochs=20)
    print("seq2seq MAE by horizon (riders): " + ", ".join(f"d+{h + 1} {v * 1e6:,.0f}" for h, v in enumerate(per_h) if h in (0, 1, 6, 13)))

    # ---- more inputs ----
    mv = multivariate_vs_univariate(data, window=56, epochs=20)
    print(f"LSTM on rail only {mv['univariate'] * 1e6:,.0f} riders; with bus + day type {mv['multivariate'] * 1e6:,.0f} riders")

    # ---- ship ----
    Xtr, ytr = make_windows(train, 56)
    Xva, yva = make_windows(np.concatenate([train[-56:], val]), 56)
    torch.manual_seed(0)
    final = RNNForecaster(32, "lstm")
    train_forecaster(final, Xtr, ytr, Xva, yva, epochs=20)
    path = os.path.join(tempfile.gettempdir(), "rail_forecaster.pt")
    save_forecaster(final, {"kind": "rnn", "hidden": 32, "cell": "lstm"}, path)
    served = load_forecaster(path)
    week = forecast_next_days(served, val[-56:], 7)
    print(f"forecast for the 7 days after {val_dates[-1]}: " + ", ".join(f"{v:,}" for v in week))


if __name__ == "__main__":
    main()

