"""
Time-Series Forecasting in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_ridership
import csv
from datetime import datetime
from pathlib import Path
import tarfile
import tempfile
import urllib.request
import numpy as np


def load_ridership():
    url = "https://github.com/ageron/data/raw/main/ridership.tgz"
    cache_path = Path(tempfile.gettempdir()) / "ridership.tgz"

    # 1. Cached tgz download
    if not cache_path.exists():
        urllib.request.urlretrieve(url, cache_path)

    # 2. Extract and parse the CSV inside the tar archive
    records = []
    seen_dates = set()

    with tarfile.open(cache_path, "r:gz") as tar:
        # Locate the CSV file inside the tarball
        csv_member = next(
            m for m in tar.getmembers() if m.name.endswith(".csv")
        )
        csv_file = tar.extractfile(csv_member)
        text_stream = (line.decode("utf-8") for line in csv_file)
        reader = csv.DictReader(text_stream)

        for row in reader:
            # 3. Convert service_date (MM/DD/YYYY) to ISO (YYYY-MM-DD)
            iso_date = datetime.strptime(
                row["service_date"], "%m/%d/%Y"
            ).strftime("%Y-%m-%d")

            # 4. Deduplicate dates, keeping the first occurrence
            if iso_date not in seen_dates:
                seen_dates.add(iso_date)
                records.append({
                    "date": iso_date,
                    "day_type": row["day_type"],
                    "bus": float(row["bus"]),
                    "rail": float(row["rail_boardings"]),
                })

    # 5. Sort chronologically by ISO date
    records.sort(key=lambda r: r["date"])

    # 6. Format output dictionary
    return {
        "dates": [r["date"] for r in records],
        "day_type": [r["day_type"] for r in records],
        "bus": np.array([r["bus"] for r in records], dtype=np.float64),
        "rail": np.array([r["rail"] for r in records], dtype=np.float64),
    }

# Step 2 - select_range
import bisect
import numpy as np


def select_range(data, start, end):
    """Cut the rail series between two ISO dates (inclusive) and rescale to millions.

    Parameters
    ----------
    data : dict
        Dictionary returned by load_ridership() with 'dates' and 'rail'.
    start : str
        Start date in ISO format (YYYY-MM-DD), inclusive.
    end : str
        End date in ISO format (YYYY-MM-DD), inclusive.

    Returns
    -------
    tuple of (list, np.ndarray)
        (dates, values) where values is rail boardings divided by 1e6.

    Raises
    ------
    ValueError
        If no dates fall within the range [start, end].
    """
    dates = data["dates"]

    # Since ISO strings are lexicographically sorted, binary search yields O(log N) slicing
    i = bisect.bisect_left(dates, start)
    j = bisect.bisect_right(dates, end)

    selected_dates = dates[i:j]

    if not selected_dates:
        raise ValueError(f"No records found for range: {start} to {end}")

    selected_rail = data["rail"][i:j] / 1e6

    return selected_dates, np.asarray(selected_rail, dtype=np.float64)

# Step 3 - naive_mae
def naive_mae(series, lag):
    # TODO: MAE of the lag-day naive forecast
    return np.abs(series[lag:] - series[:-lag]).mean()

def mae(pred, target):
    # TODO: mean absolute error as a float (accept arrays or tensors)
    pred, target = np.asarray(pred), np.asarray(target)
    return np.abs(pred-target).mean()

# Step 4 - make_windows
import torch


def make_windows(series, window):
    series = torch.as_tensor(series, dtype=torch.float32)
    n = len(series) - window
    if n <= 0:
        raise ValueError(
            f"Series length ({len(series)}) must be greater than window size ({window})."
        )

    # unfold(dimension, size, step) creates sliding windows with zero-copy views
    windows = series.unfold(dimension=0, size=window + 1, step=1)
    X = windows[:, :window]
    y = windows[:, -1]

    return X, y

# Step 5 - LinearForecaster
class LinearForecaster(nn.Module):
    def __init__(self, window):
        super().__init__()
        # TODO
        self.fc = nn.Linear(window, 1)

    def forward(self, x):
        # TODO: (batch,)
        return self.fc(x).squeeze(-1)

# Step 6 - RNNForecaster
class RNNForecaster(nn.Module):
    def __init__(self, hidden=32, cell="rnn", n_features=1):
        super().__init__()
        # TODO: self.rnn (RNN/GRU/LSTM, batch_first) and self.head
        if cell == "rnn":
            self.rnn = nn.RNN(n_features, hidden, batch_first=True)
        elif cell == "gru":
            self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        elif cell == "lstm":
            self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        else:
            raise ValueError
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # TODO: add feature axis if needed; last-step output -> head -> (batch,)
        if x.dim() ==2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)

# Step 7 - train_forecaster
import torch
import torch.nn as nn
import torch.optim as optim


def train_forecaster(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    lr=0.01,
    batch_size=32,
    seed=42,
):
    # Set seed for reproducible training
    torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    n_samples = len(X_train)
    history = {"val_mae": []}

    for epoch in range(epochs):
        # 1. Training mode & epoch-level shuffle
        model.train()
        perm = torch.randperm(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = mse_loss_fn(predictions.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()

        # 2. Validation mode (no gradient computation)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_mae = mae_loss_fn(val_preds.squeeze(-1), y_val).item()
            history["val_mae"].append(val_mae)

    return history

# Step 8 - compare_forecasters
def compare_forecasters(train_series, val_series, window=56, epochs=20, seed=42):
    # TODO: windows once; naive_1/naive_7; linear, rnn, gru, lstm trained identically; MAE dict.
    torch.manual_seed(seed)
    X_train, y_train = make_windows(train_series, window)
    X_val, y_val = make_windows(val_series, window)
    
    out = {}
    out["linear"] = train_forecaster(LinearForecaster(window=window), X_train, y_train, X_val, y_val, epochs=epochs, seed=seed)["val_mae"][-1]
    out["rnn"] = train_forecaster(RNNForecaster(), X_train, y_train, X_val, y_val, epochs=epochs, seed=seed)["val_mae"][-1]
    out["gru"] = train_forecaster(RNNForecaster(cell="gru"), X_train, y_train, X_val, y_val, epochs=epochs, seed=seed)["val_mae"][-1]
    out["lstm"] = train_forecaster(RNNForecaster(cell="lstm"), X_train, y_train, X_val, y_val, epochs=epochs, seed=seed)["val_mae"][-1]
    out["naive_1"] = naive_mae(train_series, 1)
    out["naive_7"] = naive_mae(train_series, 7)
    return out

# Step 9 - forecast_recursive
import torch


def forecast_recursive(model, last_window, steps):
    """Autoregressively roll a one-step forecasting model forward.

    Parameters
    ----------
    model : nn.Module
        Trained model expecting input shape (batch_size, window_size)
        or (batch_size, window_size, 1).
    last_window : torch.Tensor
        1D tensor of shape (window_size,) containing the most recent history.
    steps : int
        Number of steps into the future to forecast.

    Returns
    -------
    torch.Tensor
        1D float32 tensor of shape (steps,) containing the predictions.
    """
    model.eval()

    # Ensure 1D float32 clone to avoid mutating input history
    window = torch.as_tensor(last_window, dtype=torch.float32).clone().view(-1)
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            # Pass (1, window) into model, squeeze output to scalar
            pred = model(window.unsqueeze(0)).squeeze()

            # Record prediction
            predictions.append(pred.item())

            # Shift window left and append the latest forecast at the end
            window = torch.cat([window[1:], pred.view(1)])

    return torch.tensor(predictions, dtype=torch.float32)

