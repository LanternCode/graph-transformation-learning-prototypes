import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define MLP that learns from full matrix
import torch
import torch.nn as nn
import torch.optim as optim


class BinaryFocalLoss(nn.Module):
    """ Focal loss to handle imbalanced classes in binary classification. """
    def __init__(self, gamma=2, alpha=0.25):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Prevents easy examples from dominating loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


class FullMatrixMLP(nn.Module):
    def __init__(self, n):
        super(FullMatrixMLP, self).__init__()
        self.n = n
        self.fc1 = nn.Linear(n * n, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)  # Upscale back
        self.fc4 = nn.Linear(128, n * n)  # Output
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()  # Keeps outputs in [0,1]
        self.dropout = nn.Dropout(0.2)  # Add dropout after activations

    def forward(self, x):
        x1 = self.dropout(self.activation(self.fc1(x)))
        x2 = self.dropout(self.activation(self.fc2(x1)))
        x3 = self.dropout(self.activation(self.fc3(x2))) + x1  # Skip connection
        x4 = self.fc4(x3)
        return self.output_activation(x4)

    def enforce_symmetry(self, x):
        """ Convert flattened output back to symmetric matrix """
        n = int(np.sqrt(x.shape[-1]))  # Compute matrix size
        x_matrix = x.view(-1, n, n)  # Reshape back to 2D
        return 0.5 * (x_matrix + x_matrix.transpose(-1, -2))  # Symmetrize

    def symmetry_loss(self, predicted, target, lambda_weight=0.1):
        """ Custom loss that encourages symmetry """
        if predicted.shape != target.shape:
            print(f"Shape mismatch: predicted={predicted.shape}, target={target.shape}")

        mse_loss = nn.MSELoss()(predicted, target)  # Regular MSE loss

        # Corrected symmetry penalty: Apply batch-wise transpose
        symmetry_penalty = torch.norm(predicted - predicted.transpose(-1, -2), p=2)  # Frobenius norm of asymmetry

        return mse_loss + lambda_weight * symmetry_penalty


def generate_binary_dataset(num_samples, n, noise_prob=0.05):
    """ Generates a dataset of (binary asymmetric matrix, symmetric matrix) pairs with small noise. """
    X = []
    Y = []
    for _ in range(num_samples):
        # Generate random binary matrix (0s and 1s)
        asym_matrix = np.random.randint(0, 2, size=(n, n))  # 0 or 1 values

        # Introduce small random flips
        noise_mask = np.random.rand(n, n) < noise_prob
        asym_matrix[noise_mask] = 1 - asym_matrix[noise_mask]  # Flip values

        # Create a symmetric version
        sym_matrix = np.triu(asym_matrix)  # Keep upper triangular part
        sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))  # Mirror it

        X.append(asym_matrix.flatten())  # Flatten input
        Y.append(sym_matrix.flatten())  # Flatten symmetric output

    return np.array(X), np.array(Y)


# Training parameters
n = 10  # Matrix size
num_samples = 5000
epochs = 5000
learning_rate = 0.002

model = FullMatrixMLP(n)  # Pass matrix size

# Ensure the dataset is generated with the same `n`
X_train, Y_train = generate_binary_dataset(num_samples=num_samples, n=n)

# Convert dataset to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_train_tensor = (Y_train_tensor > 0.5).float()  # Ensure labels are 0 or 1

# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    predictions = model(X_train_tensor)

    # Reshape to (batch_size, n, n)
    predictions = predictions.view(-1, n, n)
    Y_train_tensor = Y_train_tensor.view(-1, n, n)

    # Compute Binary Cross Entropy Loss
    criterion = BinaryFocalLoss()  # Use focal loss instead of BCELoss
    loss = criterion(predictions, Y_train_tensor)

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


def check_matrix_similarity(original, predicted):
    """ Compute the Mean Absolute Error (MAE) between original and predicted matrices. """
    error = np.mean(np.abs(original - predicted))
    print(f"Mean Absolute Error (Input vs. Output): {error:.6f}")
    return error


def check_symmetry(matrix):
    """ Compute how symmetric the matrix is by comparing it to its transpose. """
    symmetry_error = np.mean(np.abs(matrix - matrix.T))  # MAE between matrix and its transpose
    print(f"Symmetry Error (Lower is Better): {symmetry_error:.6f}")
    return symmetry_error


# Inference
def run_inference(model, input_matrix, threshold=0.3):
    """ Runs inference on a binary matrix and rounds the output using a custom threshold. """
    n = model.n
    input_flat = input_matrix.flatten()

    # Convert input to PyTorch tensor
    input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_flat = model(input_tensor).detach().numpy().squeeze(0)

    # Reshape back into matrix format
    predicted_symmetric = predicted_flat.reshape(n, n)

    # **Apply thresholding instead of strict rounding**
    predicted_symmetric = (predicted_symmetric > threshold).astype(int)

    # **Check correctness and symmetry**
    check_matrix_similarity(input_matrix, predicted_symmetric)
    check_symmetry(predicted_symmetric)

    return predicted_symmetric



# Example test
# Generate a random test matrix
test_matrix = np.random.rand(n, n)

# Run inference and check correctness
predicted_symmetric_matrix = run_inference(model, test_matrix)

