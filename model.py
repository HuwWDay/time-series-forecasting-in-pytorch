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
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # TODO: add feature axis if needed; last-step output -> head -> (batch,)
        if x.dim() ==2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)

