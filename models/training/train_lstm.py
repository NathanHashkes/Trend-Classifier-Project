import pandas as pd
import torch
import torch.nn as nn
from data_io import choose_file
from torch.utils.data import DataLoader, Dataset
import config


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_CUTOFF_DATE = "2023-12-31"
DATA_PATH = choose_file()

INPUT_WINDOW = config.ANOMALY_BASELINE_WINDOW
HORIZON = max(config.TREND_LENGTH)


# --- LSTM ---
class LSTMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, horizon=60):
        super().__init__()
        # Process the time series
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # Output layer --> horizon vector
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # X input: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Window last day. contains all hidden State
        last_hidden_state = lstm_out[:, -1, :]
        # Send to output layer
        out = self.fc(last_hidden_state)
        return out



class StockWindowDataset(Dataset):

    def __init__(self, returns_array, input_window=INPUT_WINDOW, horizon=HORIZON):
        self.returns = returns_array
        self.input_window = input_window
        self.horizon = horizon

    def __len__(self):
        return len(self.returns) - self.input_window - self.horizon + 1

    def __getitem__(self, idx):
        x = self.returns[idx : idx + self.input_window]
        y = self.returns[idx + self.input_window : idx + self.input_window + self.horizon]

        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)


def train():

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    # filter train set by date to avoid data leakage
    df_train = df[df["date"] <= TRAIN_CUTOFF_DATE]
    # log_returns array
    returns_array = df_train["log_return_1d"].to_numpy(dtype=float)

    # DataLoader
    dataset = StockWindowDataset(returns_array, input_window=INPUT_WINDOW, horizon=HORIZON)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model object, loss function, Optimizer
    model = LSTMModel(input_size=1, hidden_size=64, horizon=HORIZON)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()  # Reset derivatives
            predictions = model(batch_x)  # Forward Pass
            loss = criterion(predictions, batch_y)  # Calculate loss
            loss.backward()  # Backward Pass
            optimizer.step()  # update weights

            total_loss += loss.item()
        #break
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

    # saving trained weights to disk
    torch.save(model.state_dict(), "lstm_model.pt")
    print("Training finished! Model saved as 'lstm_model.pt'.")


if __name__ == "__main__":
    train()