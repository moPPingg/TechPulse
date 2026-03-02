import torch
import torch.nn as nn

class iTransformerModel(nn.Module):
    """
    iTransformer architecture tailored for Green Dragon Action Score generation.
    Inverts the dimension: embedding variables into separate tokens over the time dimension.
    Provides Action Score ∈ [0,1].
    """
    def __init__(self, seq_len: int, num_features: int, d_model: int = 64, n_heads: int = 4, e_layers: int = 2, dropout: float = 0.1):
        super(iTransformerModel, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        
        # Linear projection for each variate's time series
        self.linear_emb = nn.Linear(seq_len, d_model)
        
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # Output projection back to probability
        # We flatten the representations of all variables and predict 1 target
        self.proj = nn.Linear(d_model * num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, num_features)
        """
        # Invert: shape becomes (batch_size, num_features, seq_len)
        x_inv = x.transpose(1, 2)
        
        # Embed variables across temporal dimension
        emb = self.linear_emb(x_inv)  # (batch_size, num_features, d_model)
        
        # Apply self-attention across variables
        out = self.encoder(emb)  # (batch_size, num_features, d_model)
        
        # Flatten and predict trend probability
        out = out.reshape(out.size(0), -1)  # (batch_size, num_features * d_model)
        out = self.proj(out)
        return self.sigmoid(out)
