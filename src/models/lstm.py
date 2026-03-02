import torch
import torch.nn as nn
from typing import Tuple

class LSTMModel(nn.Module):
    """
    Standard LSTM architecture for the Green Dragon Trading System.
    Outputs calibrated probability Action Score ∈ [0,1].
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, features)
        Returns:
            Probability of positive class (batch_size, 1)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
