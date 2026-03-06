import torch
from src.models.lstm import LSTMModel
import os

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 7 features: open, high, low, close, volume, ls_binary, ls_strength
model = LSTMModel(input_size=7, hidden_size=64, num_layers=2)
model.eval()

# Save the PyTorch model state dict
model_path = "models/best_lstm_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Generated and saved {model_path} successfully!")
