import torch
from torch import nn


class SymmetricClosureMLP(nn.Module):
    def __init__(self, hidden_dim=8):
        super(SymmetricClosureMLP, self).__init__()
        # Define a simple MLP: input dimension 2, one hidden layer, and output dimension 1.
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, 1, H, W)
        # Create two channels: one with the original matrix, and one with its transpose.
        x_t = x.transpose(-2, -1)
        x_cat = torch.cat([x, x_t], dim=1)  # shape: (batch, 2, H, W)

        # Reshape to apply the MLP element-wise.
        batch, channels, H, W = x_cat.shape  # channels should be 2
        # Reshape to (batch, H*W, 2)
        x_cat = x_cat.view(batch, 2, H * W).permute(0, 2, 1)

        # Apply the MLP to each 2-dimensional vector.
        out = self.mlp(x_cat)  # shape: (batch, H*W, 1)

        # Reshape back to (batch, 1, H, W)
        out = out.view(batch, H, W).unsqueeze(1)
        # Map to [0, 1] with sigmoid (later threshold at 0.5)
        return torch.sigmoid(out)
