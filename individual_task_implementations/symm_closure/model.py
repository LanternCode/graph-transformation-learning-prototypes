import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np


# --- Utility functions ---
def symmetric_closure(A):
    """
    Compute the symmetric closure of a binary matrix.

    Args:
        A: Binary matrix whose final two dimensions represent directed adjacency
            entries.

    Returns:
        Binary matrix with an entry set when either A[i, j] or A[j, i] is set.
    """
    return ((A + A.transpose(-2, -1)) > 0).float()


# --- Create training data: all 2x2 binary matrices ---
def generate_all_2x2_matrices():
    """
    Generate the complete 2x2 binary truth table for symmetric closure.

    Args:
        None.

    Returns:
        A tuple of two lists. The first list contains all 2x2 binary input
        matrices, and the second list contains their symmetric-closure targets.
    """
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
    """
    Dataset wrapper for binary matrices and symmetric-closure targets.

    The dataset stores matrix-target pairs and adds the channel dimension needed
    by the MLP training loop when individual samples are retrieved.
    """

    def __init__(self, matrices, targets):
        """
        Initialize the dataset with input matrices and target matrices.

        Args:
            matrices: Sequence of binary input matrices.
            targets: Sequence of symmetric-closure target matrices aligned with
                matrices.

        Returns:
            None.
        """
        self.matrices = matrices
        self.targets = targets

    def __len__(self):
        """
        Return the number of matrix-target pairs in the dataset.

        Args:
            None.

        Returns:
            Number of stored samples.
        """
        return len(self.matrices)

    def __getitem__(self, idx):
        """
        Fetch one matrix-target pair with a channel dimension added.

        Args:
            idx: Integer index of the sample to retrieve.

        Returns:
            Tuple containing the input tensor and target tensor, each shaped as
            (1, H, W).
        """
        x = self.matrices[idx].clone()
        y = self.targets[idx].clone()
        # Add channel dimension: shape becomes (1, H, W)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y


# --- Define the MLP model ---
class SymmetricClosureMLP(nn.Module):
    """
    Pointwise MLP for learning the symmetric-closure rule.

    The model scores each matrix entry using the pair consisting of the original
    entry and its transposed counterpart.
    """

    def __init__(self, hidden_dim=8):
        """
        Initialize the symmetric-closure MLP.

        Args:
            hidden_dim: Width of the hidden layer used for each pointwise
                two-value input.

        Returns:
            None.
        """
        super().__init__()
        # Define a simple MLP: input dimension 2, one hidden layer, and output dimension 1.
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Compute symmetric-closure probabilities for an input matrix batch.

        Args:
            x: Input tensor shaped as (batch, 1, H, W).

        Returns:
            Tensor of probabilities shaped as (batch, 1, H, W).
        """
        # x: (batch, 1, H, W)
        # Create two channels: one with the original matrix, and one with its transpose.
        x_t = x.transpose(-2, -1)
        x_cat = torch.cat([x, x_t], dim=1)  # shape: (batch, 2, H, W)

        # Reshape to apply the MLP element-wise.
        batch, _, H, W = x_cat.shape
        # Reshape to (batch, H*W, 2)
        x_cat = x_cat.view(batch, 2, H * W).permute(0, 2, 1)

        # Apply the MLP to each 2-dimensional vector.
        out = self.mlp(x_cat)  # shape: (batch, H*W, 1)

        # Reshape back to (batch, 1, H, W)
        out = out.view(batch, H, W).unsqueeze(1)
        # Map to [0, 1] with sigmoid (later threshold at 0.5)
        return torch.sigmoid(out)


# --- Testing on n x n matrices ---
def generate_random_n_by_n_binary_matrix(num_matrices, n):
    """
    Generate random n-by-n binary matrices for model evaluation.

    Args:
        num_matrices: Number of random matrices to generate.
        n: Height and width of each square binary matrix.

    Returns:
        List of randomly generated float tensors shaped as (n, n).
    """
    samples = []
    for _ in range(num_matrices):
        mat = torch.randint(0, 2, (n, n)).float()
        samples.append(mat)
    return samples


if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.1

    # Generate training data (2x2 matrices)
    train_mats, train_targets = generate_all_2x2_matrices()
    train_dataset = GraphDataset(train_mats, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize model and optimizer
    model = SymmetricClosureMLP()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # --- Training loop ---
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)  # shape: (batch, 1, H, W)
            # Compute per-entry error.
            error = pred - y
            # Backpropagate using the error directly.
            error.backward(gradient=error)
            optimizer.step()
            batch_loss = error.abs().mean().item()
            epoch_loss += batch_loss
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Mean Absolute Error: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "symmetric_closure_mlp.pth")

    # Evaluate the model on the test samples
    total_error_count = 0
    total_elements = 0  # To count all possible guesses
    model.eval()
    test_samples = generate_random_n_by_n_binary_matrix(1000, 1000)
    with torch.no_grad():
        for test_mat in test_samples:
            expected = symmetric_closure(test_mat)
            # Prepare input: add batch and channel dimensions -> shape (1, 1, n, n)
            test_input = test_mat.unsqueeze(0).unsqueeze(0)
            pred = model(test_input).squeeze(0).squeeze(0)  # shape: (n, n)
            # Threshold prediction at 0.5 to obtain a binary output
            pred_binary = (pred > 0.5).float()
            error_matrix = pred_binary - expected
            error_count = np.count_nonzero(error_matrix)

            total_error_count += error_count
            total_elements += expected.numel()

    total_correct_count = total_elements - total_error_count

    print(f"\nTotal number of errors over all test samples: {total_error_count}")
    print(f"Total number of correct guesses: {total_correct_count}")
    print(f"Accuracy: {(total_correct_count / total_elements) * 100:.2f}%")
