import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data_io import choose_file, load_file
from dataset import StockWindowDataset
from trainer import train_model, export_and_save_model

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_CUTOFF_DATE = "2023-12-31"


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, horizon: int):
        super().__init__()
        # input_size receives the number of features
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the hidden state from the last step in the window
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)  # shape: (batch_size, horizon)
        return out


def main():
    # Load data
    path = choose_file()
    df = load_file(path, config.REQUIRED_MODEL_COLUMNS)

    # Filter by date to prevent data leakage
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] <= pd.Timestamp(TRAIN_CUTOFF_DATE)].copy()

    # only for basic check if runs
    #train_df = train_df.head(5000)

    # Prepare Dataset and DataLoader
    dataset = StockWindowDataset(
        df=train_df,
        feature_columns=config.MODEL_FEATURE_COLUMNS,
        target_column="log_return_1d",
        input_window=config.MODEL_INPUT_SIZE,
        horizon=max(config.TREND_LENGTH)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize and train the model
    model = LSTMModel(
        input_size=len(config.MODEL_FEATURE_COLUMNS),
        hidden_size=64,
        horizon=max(config.TREND_LENGTH)
    )

    trained_model = train_model(
        model=model,
        dataloader=dataloader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Export and save in pt2 format compatible with model.py
    export_and_save_model(
        model=trained_model,
        save_filename="lstm_model.pt2"
    )


if __name__ == "__main__":
    main()
