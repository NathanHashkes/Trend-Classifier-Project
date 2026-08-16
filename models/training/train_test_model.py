from pathlib import Path
import torch
import torch.nn as nn
import config

MODEL_NAME = "dummy_model.pt2"

class DummyModel(nn.Module):
    def __init__(self, output_length):
        super().__init__()
        self.output_length = output_length

    def forward(self, x):
        # Take the most recent return
        last_return = x[:, -1, 0].unsqueeze(1)

        # Pretend every future day will have that same return
        return last_return.repeat(1, self.output_length)

output_length = max(config.TREND_LENGTH)
model = DummyModel(output_length)
model.eval()

# Example of the input format our models receive:
example_input = torch.zeros(1, config.MODEL_INPUT_SIZE, len(config.MODEL_FEATURE_COLUMNS))

exported_program = torch.export.export(model,(example_input,))

save_path = Path(config.MODEL_SAVE_PATH) / MODEL_NAME

torch.export.save(exported_program, save_path)

print(f"Dummy model saved to {save_path}")