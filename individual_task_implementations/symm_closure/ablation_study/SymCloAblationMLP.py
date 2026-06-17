"""
Train and evaluate an MLP ablation without transpose access.

This prototype removes the paired transpose feature used by the full symmetric
closure MLP. The model processes each scalar A[i, j] independently, so it tests
whether a pointwise MLP can recover A OR Aᵀ without direct access to A[j, i].
"""
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np


def symmetric_closure(A):
    # Ground-truth: each element is 1 if A[i,j] or A[j,i] is 1.
    return ((A + A.transpose(-2, -1)) > 0).float()


def generate_all_2x2_matrices():
    mats = []
    targets = []
    for bits in itertools.product([0, 1], repeat=4):
        mat = torch.tensor(bits, dtype=torch.float32).view(2, 2)
        target = symmetric_closure(mat)
        mats.append(mat)
        targets.append(target)
    return mats, targets


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, targets):
        self.matrices = matrices
        self.targets = targets

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        x = self.matrices[idx].clone()
        y = self.targets[idx].clone()
        # Add channel dimension so that x and y have shape [1, H, W]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y


class SymmetricClosureMLP(nn.Module):
    def __init__(self, hidden_size=8):
        super(SymmetricClosureMLP, self).__init__()
        # Define an MLP that maps a scalar to a scalar.
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, 1, n, n]
        # Instead of using the transposed input, operate on x directly.
        x_flat = x.reshape(-1, 1)  # Flatten to shape [batch * n * n, 1]
        hidden = torch.relu(self.fc1(x_flat))
        out_flat = self.fc2(hidden)
        # Reshape back to [batch, 1, n, n]
        out = out_flat.reshape(x.size())
        return torch.sigmoid(out)


# Training parameters
num_epochs = 1000
learning_rate = 0.1

# Generate training data for 2x2 matrices.
train_mats, train_targets = generate_all_2x2_matrices()
train_dataset = GraphDataset(train_mats, train_targets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize the MLP model (now operating directly on x).
model = SymmetricClosureMLP(hidden_size=8)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        error = pred - y
        error.backward(gradient=error)
        optimizer.step()
        batch_loss = error.abs().mean().item()
        epoch_loss += batch_loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Mean Abs Error: {epoch_loss / len(train_loader):.4f}")


def generate_random_n_by_n_binary_matrix(num_matrices, n):
    samples = []
    for _ in range(num_matrices):
        mat = torch.randint(0, 2, (n, n)).float()
        samples.append(mat)
    return samples


# Test on arbitrary size matrices (e.g. 100x100).
test_samples = generate_random_n_by_n_binary_matrix(8000, 10)

total_error_count = 0
total_elements = 0
model.eval()
with torch.no_grad():
    for test_mat in test_samples:
        expected = symmetric_closure(test_mat)
        test_input = test_mat.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, n, n]
        pred = model(test_input).squeeze(0).squeeze(0)  # Shape: [n, n]
        pred_binary = (pred > 0.5).float()
        error_matrix = pred_binary - expected
        error_count = np.count_nonzero(error_matrix)
        total_error_count += error_count
        total_elements += expected.numel()

total_correct_count = total_elements - total_error_count

print(f"\nTotal number of errors over all test samples: {total_error_count}")
print(f"Total number of correct guesses: {total_correct_count}")
print(f"Accuracy: {(total_correct_count / total_elements) * 100:.2f}%")
