"""
Train and evaluate a CNN-style symmetric closure model.

This prototype learns the Boolean symmetric closure operation A OR Aᵀ from all
2x2 binary matrices, then tests whether the learned local rule generalizes to
larger binary matrices. The model receives both the matrix and its transpose as
two input channels, and applies a shared 1x1 convolution to predict each output
entry independently.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np


# --- Utility functions ---
def symmetric_closure(A):
    """
    Compute the symmetric closure (OR) of a binary matrix.
    For binary values, A OR Aᵀ is equivalent to (A + Aᵀ > 0).
    """
    return ((A + A.transpose(-2, -1)) > 0).float()


# --- Create training data: all 2x2 binary matrices ---
def generate_all_2x2_matrices():
    mats = []
    targets = []
    for bits in itertools.product([0, 1], repeat=4):
        mat = torch.tensor(bits, dtype=torch.float32).view(2, 2)
        target = symmetric_closure(mat)
        mats.append(mat)
        targets.append(target)
    return mats, targets


# --- PyTorch Dataset ---
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, targets):
        self.matrices = matrices
        self.targets = targets

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        x = self.matrices[idx].clone()
        y = self.targets[idx].clone()
        # Add channel dimension: shape becomes (1, H, W)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y


# --- Define the model ---
class SymmetricClosureCNN(nn.Module):
    def __init__(self):
        super(SymmetricClosureCNN, self).__init__()
        # Our input is two channels: one is the matrix itself and the other its transpose.
        # A 1x1 convolution here acts like an elementwise linear layer shared across positions.
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, 1, H, W)
        # Create a two-channel input: channel 0 is x; channel 1 is x transposed (i.e. A[j,i])
        x_t = x.transpose(-2, -1)
        x_cat = torch.cat([x, x_t], dim=1)  # shape (batch, 2, H, W)
        out = self.conv(x_cat)  # shape (batch, 1, H, W)
        # Map to [0,1] with sigmoid (later threshold at 0.5)
        return torch.sigmoid(out)


# --- Training setup ---

# Hyperparameters
num_epochs = 1000
learning_rate = 0.1

# Generate training data (2x2 matrices)
train_mats, train_targets = generate_all_2x2_matrices()
train_dataset = GraphDataset(train_mats, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize model and optimizer
model = SymmetricClosureCNN()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- Training loop ---
# Here we define our "loss" per entry as (prediction - target).
# Instead of summing or averaging, we use the error matrix directly to compute gradients.
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)  # shape: (batch, 1, H, W)
        # Compute per-entry error: positive if over-predicted, negative if under-predicted.
        error = pred - y
        # Backpropagate using the error itself.
        # This is equivalent to using the derivative of 1/2*(pred - y)^2,
        # because d/dpred (1/2*(pred - y)^2) = (pred - y).
        error.backward(gradient=error)
        optimizer.step()
        batch_loss = error.abs().mean().item()
        epoch_loss += batch_loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Mean Abs Error: {epoch_loss / len(train_loader):.4f}")


# --- Testing on n x n matrices ---
def generate_random_n_by_n_binary_matrix(num_matrices, n):
    samples = []
    for _ in range(num_matrices):
        mat = torch.randint(0, 2, (n, n)).float()
        samples.append(mat)
    return samples


test_samples = generate_random_n_by_n_binary_matrix(8000, 100)

# Evaluate the model on the test samples
total_error_count = 0
total_elements = 0  # To count all possible guesses
model.eval()
with torch.no_grad():
    for i, test_mat in enumerate(test_samples):
        expected = symmetric_closure(test_mat)
        # Prepare input: add batch and channel dimensions -> shape (1, 1, 8, 8)
        test_input = test_mat.unsqueeze(0).unsqueeze(0)
        pred = model(test_input).squeeze(0).squeeze(0)  # shape (8, 8)
        # Threshold prediction at 0.5 to obtain a binary output
        pred_binary = (pred > 0.5).float()
        error_matrix = pred_binary - expected
        error_count = np.count_nonzero(error_matrix)

        # Accumulate total errors
        total_error_count += error_count

        # Accumulate total number of elements (possible guesses)
        total_elements += expected.numel()  # expected is (8,8), so numel() gives total elements


# Compute the number of correct guesses
total_correct_count = total_elements - total_error_count

# Print total errors and correct guesses
print(f"\nTotal number of errors over all test samples: {total_error_count}")
print(f"Total number of correct guesses: {total_correct_count}")
print(f"Accuracy: {(total_correct_count / total_elements)*100:.2f}%")  # Compute accuracy as well
