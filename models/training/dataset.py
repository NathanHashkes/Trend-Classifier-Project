import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config


class StockWindowDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            feature_columns: list[str] = None,
            target_column: str = "log_return_1d",
            input_window: int = config.MODEL_INPUT_SIZE,
            horizon: int = max(config.TREND_LENGTH),
            ticker_column: str = "ticker",
            date_column: str = "date"
    ):
        self.input_window = input_window
        self.horizon = horizon
        self.feature_columns = feature_columns if feature_columns is not None else config.MODEL_FEATURE_COLUMNS
        self.target_column = target_column


        required = set(self.feature_columns + [self.target_column, ticker_column, date_column])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        self.samples_x = []
        self.samples_y = []

        for ticker, group in df.groupby(ticker_column):
            sorted_group = group.sort_values(date_column).reset_index(drop=True)

            features = sorted_group[self.feature_columns].to_numpy(dtype=np.float32)
            targets = sorted_group[self.target_column].to_numpy(dtype=np.float32)

            num_rows = len(sorted_group)
            total_window = self.input_window + self.horizon

            if num_rows < total_window:
                continue

            for idx in range(num_rows - total_window + 1):
                x = features[idx: idx + self.input_window]
                y = targets[idx + self.input_window: idx + total_window]

                self.samples_x.append(x)
                self.samples_y.append(y)

        if len(self.samples_x) == 0:
            raise ValueError("No valid windows could be created from the given dataset.")

    def __len__(self) -> int:
        return len(self.samples_x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.samples_x[idx], dtype=torch.float32)  # shape: (input_window, num_features)
        y = torch.tensor(self.samples_y[idx], dtype=torch.float32)  # shape: (horizon,)
        return x, y