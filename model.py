import numpy as np
import torch
import torch.nn as nn
import config
import os


MODEL_PATH = "lstm_model.pt"

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


# Load model
_model_instance = None


def _load_trained_model(horizon):
    global _model_instance

    if _model_instance is not None:
        return _model_instance

    lstm_net = LSTMModel(input_size=1, hidden_size=64, horizon=horizon)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file '{MODEL_PATH}' not found. Run 'train_model.py' first!")

    lstm_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    lstm_net.eval()

    _model_instance = lstm_net
    return _model_instance


def get_predictions(recent_returns, horizon=None):
    if horizon is None:
        horizon = max(config.TREND_LENGTH)

    # Load model
    lstm_net = _load_trained_model(horizon=horizon)

    returns_tensor = torch.tensor(recent_returns, dtype=torch.float32).view(1, -1, 1)

    with torch.no_grad():
        predictions_tensor = lstm_net(returns_tensor)

    predicted_returns = predictions_tensor.squeeze(0).numpy()

    return predicted_returns