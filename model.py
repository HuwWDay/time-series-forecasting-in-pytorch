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

# Step 10 - make_seq2seq_windows
import torch


def make_seq2seq_windows(series, window, horizon):
    """Construct sequence-to-sequence input and target tensors.

    Parameters
    ----------
    series : array-like or torch.Tensor
        1D time series.
    window : int
        Number of input time steps in each window.
    horizon : int
        Number of future steps to predict at every step inside the window.

    Returns
    -------
    X : torch.Tensor
        Float32 tensor of shape (n, window), where X[i] = series[i : i + window].
    Y : torch.Tensor
        Float32 tensor of shape (n, window, horizon), where
        Y[i, t, h] = series[i + t + 1 + h].
    """
    s = torch.as_tensor(series, dtype=torch.float32).view(-1)
    total_len = len(s)
    n = total_len - window - horizon + 1

    if n <= 0:
        raise ValueError(
            f"Series length ({total_len}) is too short for window ({window}) "
            f"and horizon ({horizon}). Requires at least {window + horizon - 1} points."
        )

    # 1. Input windows: shape (n, window)
    # unfold(dimension, size, step) creates sliding window slices
    X = s[: n + window - 1].unfold(dimension=0, size=window, step=1)

    # 2. Target windows: shape (n, window, horizon)
    # At each time step t (0 <= t < window), target is the next `horizon` values:
    # starting from index 1 up to end of the series.
    # Total span for each target slice across `window` steps is (window + horizon - 1).
    target_spans = s[1:].unfold(
        dimension=0, size=window + horizon - 1, step=1
    )[:n]
    Y = target_spans.unfold(dimension=1, size=horizon, step=1)

    return X.contiguous(), Y.contiguous()

# Step 11 - Seq2SeqForecaster
class Seq2SeqForecaster(nn.Module):
    def __init__(self, hidden=32, horizon=14, n_features=1):
        super().__init__()
        # TODO: LSTM + Linear(hidden, horizon)
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        # TODO: (batch, window, horizon)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.head(out)

# Step 12 - train_seq2seq
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_seq2seq(
    model, X, Y, X_val, Y_val, epochs=20, lr=0.01, batch_size=32, seed=42
):
    """Train a sequence-to-sequence model using MSE over all sequence steps,

    and evaluate the per-horizon MAE at the final time step on the validation
    set.

    Parameters
    ----------
    model : nn.Module
        Seq2Seq model outputting predictions of shape (batch, window, horizon).
    X : torch.Tensor
        Training inputs of shape (n_train, window).
    Y : torch.Tensor
        Training targets of shape (n_train, window, horizon).
    X_val : torch.Tensor
        Validation inputs of shape (n_val, window).
    Y_val : torch.Tensor
        Validation targets of shape (n_val, window, horizon).
    epochs : int, default=20
        Number of full passes over the training set.
    lr : float, default=0.01
        Learning rate for the Adam optimizer.
    batch_size : int, default=32
        Mini-batch size.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    list of float
        Validation MAE for each horizon step h = 0, ..., horizon - 1,
        evaluated solely on the last step of the window ([:, -1, h]).
    """
    torch.manual_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = len(X)

    # 1. Training loop over all steps and horizons
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = perm[start_idx : start_idx + batch_size]
            X_batch = X[batch_idx]
            Y_batch = Y[batch_idx]

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = F.mse_loss(preds, Y_batch)
            loss.backward()
            optimizer.step()

    # 2. Evaluation: score the last time step per horizon
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val)  # Shape: (n_val, window, horizon)

        # Slice the last time step inside the window: shape (n_val, horizon)
        val_preds_last = val_preds[:, -1, :]
        Y_val_last = Y_val[:, -1, :]

        # Compute mean absolute error across batches for each horizon step h
        # abs_errors shape: (n_val, horizon) -> mean over dim 0 -> (horizon,)
        per_horizon_mae = (
            torch.mean(torch.abs(val_preds_last - Y_val_last), dim=0)
            .cpu()
            .tolist()
        )

    return per_horizon_mae

# Step 13 - feature_matrix
import bisect
import numpy as np
import torch


def feature_matrix(data, start, end):
    """Build a multivariate daily feature table for the ISO date range (inclusive).

    Columns: [rail / 1e6, bus / 1e6, is_W, is_A, is_U]

    Parameters
    ----------
    data : dict
        Dict containing 'dates', 'day_type', 'bus', and 'rail'.
    start : str
        Start date in ISO format (YYYY-MM-DD), inclusive.
    end : str
        End date in ISO format (YYYY-MM-DD), inclusive.

    Returns
    -------
    np.ndarray
        Array of shape (n, 5) with dtype np.float64.
    """
    dates = data["dates"]
    i = bisect.bisect_left(dates, start)
    j = bisect.bisect_right(dates, end)

    if i >= j:
        raise ValueError(f"No records found for range: {start} to {end}")

    # Slice components across the selected range
    rail_slice = data["rail"][i:j] / 1e6
    bus_slice = data["bus"][i:j] / 1e6
    day_types = np.array(data["day_type"][i:j])

    # One-hot encode day_type ('W' = Weekday, 'A' = Saturday, 'U' = Sunday/Holiday)
    is_w = (day_types == "W").astype(np.float64)
    is_a = (day_types == "A").astype(np.float64)
    is_u = (day_types == "U").astype(np.float64)

    # Stack columns to shape (n, 5)
    return np.column_stack([rail_slice, bus_slice, is_w, is_a, is_u]).astype(
        np.float64
    )


def make_multivariate_windows(feats, window):
    """Construct multivariate sliding windows for one-step-ahead rail forecasting.

    Parameters
    ----------
    feats : np.ndarray or torch.Tensor
        Array of shape (n_days, 5).
    window : int
        Window size (number of past days).

    Returns
    -------
    X : torch.Tensor
        Float32 tensor of shape (n, window, 5).
    y : torch.Tensor
        Float32 tensor of shape (n,) containing next-day rail values (column 0).
    """
    feats = np.asarray(feats, dtype=np.float32)
    n = len(feats) - window
    if n <= 0:
        raise ValueError(
            f"Series length ({len(feats)}) must be greater than window size ({window})."
        )

    X = np.empty((n, window, feats.shape[1]), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)

    for i in range(n):
        X[i] = feats[i : i + window]
        y[i] = feats[i + window, 0]  # Target is next-day rail (col 0)

    return torch.from_numpy(X), torch.from_numpy(y)

# Step 14 - multivariate_vs_univariate
# ── Step 014  multivariate_vs_univariate ──
def multivariate_vs_univariate(data, window=56, epochs=20, seed=42):
    """Compare univariate (rail only) vs multivariate (rail + bus + day_type) LSTM models.

    Parameters
    ----------
    data : dict
        Dataset dict returned by load_ridership().
    window : int, default=56
        Past observation window size.
    epochs : int, default=20
        Number of training epochs.
    seed : int, default=42
        Reproducibility seed.

    Returns
    -------
    dict
        {"univariate": mae, "multivariate": mae} validation MAE in millions of riders.
    """
    train_start, train_end = "2016-01-01", "2018-12-31"
    val_start, val_end = "2019-01-01", "2019-05-31"

    # 1. Prepare univariate rail series
    _, rail_train = select_range(data, train_start, train_end)
    _, rail_val = select_range(data, val_start, val_end)
    rail_val_extended = np.concatenate([rail_train[-window:], rail_val])

    X_train_uni, y_train_uni = make_windows(rail_train, window)
    X_val_uni, y_val_uni = make_windows(rail_val_extended, window)

    # 2. Prepare multivariate feature matrices
    feats_train = feature_matrix(data, train_start, train_end)
    feats_val = feature_matrix(data, val_start, val_end)
    feats_val_extended = np.vstack([feats_train[-window:], feats_val])

    X_train_multi, y_train_multi = make_multivariate_windows(
        feats_train, window
    )
    X_val_multi, y_val_multi = make_multivariate_windows(
        feats_val_extended, window
    )

    # 3. Train univariate LSTM
    torch.manual_seed(seed)
    uni_model = RNNForecaster(hidden=32, cell="lstm", n_features=1)
    uni_history = train_forecaster(
        uni_model,
        X_train_uni,
        y_train_uni,
        X_val_uni,
        y_val_uni,
        epochs=epochs,
        lr=0.01,
        seed=seed,
    )

    # 4. Train multivariate LSTM
    torch.manual_seed(seed)
    multi_model = RNNForecaster(hidden=32, cell="lstm", n_features=5)
    multi_history = train_forecaster(
        multi_model,
        X_train_multi,
        y_train_multi,
        X_val_multi,
        y_val_multi,
        epochs=epochs,
        lr=0.01,
        seed=seed,
    )

    return {
        "univariate": float(uni_history["val_mae"][-1]),
        "multivariate": float(multi_history["val_mae"][-1]),
    }

# Step 15 - save_forecaster
# ── Step 015  save_forecaster ──
def save_forecaster(model, config, path):
    """Serialize the model state dictionary along with its construction metadata.

    Parameters
    ----------
    model : nn.Module
        The trained forecaster model.
    config : dict
        Configuration dictionary (e.g., {"kind": "rnn", "hidden": 32, "cell": "lstm"}).
    path : str or Path
        Destination file path.
    """
    checkpoint = {
        "config": config,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def load_forecaster(path):
    """Load model metadata and weights from disk, reconstructing the instance.

    Parameters
    ----------
    path : str or Path
        File path to the serialized checkpoint.

    Returns
    -------
    nn.Module
        Reconstructed model loaded with the saved weights and set to eval mode.
    """
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint["config"]

    kind = config.get("kind", "rnn")
    if kind == "linear":
        model = LinearForecaster(window=config["window"])
    elif kind == "rnn":
        model = RNNForecaster(
            hidden=config.get("hidden", 32),
            cell=config.get("cell", "rnn"),
            n_features=config.get("n_features", 1),
        )
    elif kind == "seq2seq":
        model = Seq2SeqForecaster(
            hidden=config.get("hidden", 32),
            horizon=config.get("horizon", 14),
            n_features=config.get("n_features", 1),
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

