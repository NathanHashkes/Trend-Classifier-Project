from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    criterion: nn.Module = None,
    device: str = "cpu"
) -> nn.Module:

    if criterion is None:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

    return model



def export_and_save_model(
    model: nn.Module,
    save_filename: str,
    input_window: int = config.MODEL_INPUT_SIZE,
    num_features: int = len(config.MODEL_FEATURE_COLUMNS),
    save_dir: str = config.MODEL_SAVE_PATH
) -> Path:
    model.eval()
    model.to("cpu")

    example_input = torch.zeros(1, input_window, num_features, dtype=torch.float32)

    # saving via torch.export
    exported_program = torch.export.export(model, (example_input,))

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / save_filename

    torch.export.save(exported_program, save_path)
    print(f"Model successfully exported and saved to {save_path}")
    return save_path
