import torch
import torch.nn as nn

class PatchTSTModel(nn.Module):
    """
    PatchTST architecture implemented for the Green Dragon Trading System.
    Segments sequences into overlapping patches and models dependencies.
    Outputs calibrated Action Score Action Score ∈ [0,1].
    """
    def __init__(self, seq_len: int, num_features: int, patch_len: int = 16, stride: int = 8, 
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 2, dropout: float = 0.1):
        super(PatchTSTModel, self).__init__()
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        # Assuming padding adds 'stride' length to sequence:
        self.patch_num = int((seq_len + stride - patch_len) / stride + 1)
        
        self.linear_patch = nn.Linear(patch_len, d_model)
        
        # Position embedding over patches
        self.w_pos = nn.Parameter(torch.empty(1, self.num_features, self.patch_num, d_model))
        nn.init.uniform_(self.w_pos, -0.02, 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # Final head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, num_features)
        """
        batch_size = x.size(0)
        
        # Pad sequence
        x = x.transpose(1, 2)  # (batch_size, num_features, seq_len)
        x = self.padding_patch_layer(x)  # (batch_size, num_features, seq_len + stride)
        
        # Create patches
        # output shape: (batch_size, num_features, patch_num, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Embed patches via linear projection
        x = self.linear_patch(x)  # (batch_size, num_features, patch_num, d_model)
        
        # Add position embeddings
        x = x + self.w_pos
        x = x.reshape(-1, self.patch_num, x.size(-1))  # (batch_size * num_features, patch_num, d_model)
        
        # Apply Transformer Encoder over patches
        x = self.encoder(x)  # (batch_size * num_features, patch_num, d_model)
        
        # Reshape back to independent features and average/flatten
        x = x.reshape(batch_size, self.num_features, self.patch_num, -1)
        
        # For simplicity and matching common PatchTST trend prediction:
        # We average across univariates or flatten. Here we average features:
        z = x.mean(dim=1)  # (batch_size, patch_num, d_model)
        
        return self.head(z)
