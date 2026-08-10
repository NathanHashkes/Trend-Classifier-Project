from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import  ANOMALY_BASELINE_WINDOW, ANOMALY_MAX_LENGTH, ANOMALY_THRESHOLD, SCAN_DATE

@dataclass(frozen=True)
class AnomalyResult:
    """All the data about a detected anomaly."""
    length: int
    start: np.datetime64
    end: np.datetime64
    z_score: float
    window_mean: float
    baseline_mean: float
    baseline_std: float

    @property
    def direction(self) -> int:
        """returns the anomaly direction: 1 or -1"""
        return 1 if self.z_score > 0 else -1
 
def detect_anomaly(stock_data, scan_date=SCAN_DATE, baseline_window=ANOMALY_BASELINE_WINDOW, max_length=ANOMALY_MAX_LENGTH, 
                   threshold=ANOMALY_THRESHOLD, return_column="log_return_1d"):
    """Scan for an anomaly at one specific date.
 
    Gets "stock_data", which must be sorted newest to oldest.
    Returns an AnomalyResult, or None if no window crosses the threshold."""
    needed = baseline_window + max_length
 
    if scan_date is None:
        scan_data = stock_data
    else:
        scan_data = stock_data[stock_data["date"] <= pd.Timestamp(scan_date)]
 
    scan_data = scan_data.head(needed).reset_index(drop=True)
 
    if len(scan_data) < needed:
        raise ValueError(f"Only {len(scan_data)} rows available at or before {scan_date}, need {needed}")
 
    dates = scan_data["date"].to_numpy(dtype="datetime64[D]")
    returns = scan_data[return_column].to_numpy(dtype=float)
 
    if np.isnan(returns).any():
        raise ValueError(f"Missing {return_column} values in the scan range")
 
    for k in range(1, max_length + 1): # The actual calulation of the z-score is here
        window = returns[:k]
        baseline = returns[k:k + baseline_window]
 
        baseline_std = baseline.std(ddof=1)
        standard_error = baseline_std / np.sqrt(k)
 
        if standard_error == 0:
            continue
 
        z_score = (window.mean() - baseline.mean()) / standard_error
 
        if abs(z_score) > threshold:
            return AnomalyResult(
                length=k,
                start=dates[k - 1],
                end=dates[0],
                z_score=z_score,
                window_mean=window.mean(),
                baseline_mean=baseline.mean(),
                baseline_std=baseline_std,
            )
 
    return None