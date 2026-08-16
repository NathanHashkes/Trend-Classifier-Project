import torch
import config
from pathlib import Path

def get_predictions(input_data, horizon):
    """Gets a input dataframe with data that the model needs
    
        Returns predictions up to the requested horizon.
        
        WARNING: How many days the model actually predicts can be different"""
    model_path = Path(config.MODEL_SAVE_PATH) / config.MODEL_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    exported_program = torch.export.load(model_path)
    model = exported_program.module()

    feature_data = input_data[config.MODEL_FEATURE_COLUMNS]

    model_input = torch.tensor(feature_data.to_numpy(dtype="float32")).unsqueeze(0)

    with torch.no_grad():
        predictions = model(model_input)

    predictions = predictions.squeeze(0).numpy()

    if len(predictions) < horizon:
        raise ValueError(f"Model only predicts {len(predictions)} days, but {horizon} days were requested.")

    return predictions[:horizon]