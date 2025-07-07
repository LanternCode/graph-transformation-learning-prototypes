import torch.nn.functional as F
import torch

# Define BCE parameters
pred = 0.72
true = 0.0

# Convert inputs to tensors
prediction = torch.tensor([pred], dtype=torch.float32)  # Model prediction
target = torch.tensor([true], dtype=torch.float32)  # Ground truth (binary: 0 or 1)

# Compute BCE loss
bce_loss = F.binary_cross_entropy(prediction, target)

print(f"BCE Loss: {bce_loss.item()}")  # Should return a valid number

'''
BCE(0.4, 1) ≈ 0.916
BCE(0.5, 1) ≈ 0.693
BCE(0.6, 1) ≈ 0.511
BCE(0.4, 0) ≈ 0.511
BCE(0.5, 0) ≈ 0.693
BCE(0.6, 0) ≈ 0.916
'''