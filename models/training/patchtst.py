import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data_io import choose_file, load_file
from dataset import StockWindowDataset
from trainer import train_model, export_and_save_model


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
TRAIN_CUTOFF_DATE = "2023-12-31"

PATCH_LEN = 16       # Length of each Patch
STRIDE = 8           # Stride (step) between consecutive Patches
D_MODEL = 64         # Embedding dimension
NHEAD = 4            # Number of Attention heads
NUM_LAYERS = 2       # Number of Transformer Encoder layers
DROPOUT = 0.1


class PatchTST(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_features: int,
        horizon: int,
        patch_len: int,
        stride: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride

        # Calculate the number of patches created from the time series
        self.num_patches = (input_size - patch_len) // stride + 1

        # Patch Projection: convert each Patch into a Token
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection Head: projection from all concatenated Tokens to the Horizon
        self.head = nn.Linear(self.num_patches * d_model, horizon)

        # Channel Aggregation: combine channels into a one-dimensional return forecast
        self.channel_aggregator = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (Batch, Lookback, Features) -> (b, l, m)
        b, l, m = x.shape

        # Change to channel-first order: (b, m, l)
        x = x.permute(0, 2, 1)

        # Extract Patches with a sliding window: output shape (b, m, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Channel Independence: each Feature is processed as a separate sequence
        x = x.reshape(b * m, self.num_patches, self.patch_len)

        # Linear projection + Positional Embedding
        enc_in = self.patch_embedding(x) + self.pos_embedding  # (b * m, num_patches, d_model)

        # Pass through the Transformer
        enc_out = self.transformer(enc_in)  # (b * m, num_patches, d_model)

        # Flatten the Tokens
        flatten_out = enc_out.reshape(b * m, -1)  # (b * m, num_patches * d_model)

        # Projection to the Horizon
        out = self.head(flatten_out)  # (b * m, horizon)

        # Restore original channel structure: (b, m, horizon) -> (b, horizon, m)
        out = out.reshape(b, m, self.horizon).permute(0, 2, 1)

        # Combine all features into a single forecast vector
        out = self.channel_aggregator(out).squeeze(-1)  # (b, horizon)

        return out


def main():
    # Select and load the data
    path = choose_file()
    df = load_file(path, config.REQUIRED_MODEL_COLUMNS)

    # Filter dates to prevent data leakage
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[df["date"] <= pd.Timestamp(TRAIN_CUTOFF_DATE)].copy()

    # only for basic check if runs
    #train_df = train_df.head(5000)

    # Build the Dataset and DataLoader
    dataset = StockWindowDataset(
        df=train_df,
        feature_columns=config.MODEL_FEATURE_COLUMNS,
        target_column="log_return_1d",
        input_window=config.MODEL_INPUT_SIZE,
        horizon=max(config.TREND_LENGTH)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize and train the model
    model = PatchTST(
        input_size=config.MODEL_INPUT_SIZE,
        num_features=len(config.MODEL_FEATURE_COLUMNS),
        horizon=max(config.TREND_LENGTH),
        patch_len=PATCH_LEN,
        stride=STRIDE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
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
        save_filename="patchtst_model.pt2"
    )


if __name__ == "__main__":
    main()