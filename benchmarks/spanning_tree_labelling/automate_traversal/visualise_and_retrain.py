import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from test_three import ContextAwareMLP, generate_candidate_graph, get_contextual_edge_features, compute_loss


# === LOSS SURFACE VISUALIZATION ===
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        shape = p.shape
        size = p.numel()
        p.data.copy_(flat[i:i+size].view(shape))
        i += size

def visualize_loss_surface(model_path, steps=21, scale=1.0):
    model = ContextAwareMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Flatten original weights
    w0 = get_flat_params(model)

    # Generate two random directions
    d1 = torch.randn_like(w0)
    d2 = torch.randn_like(w0)
    d1 /= torch.norm(d1)
    d2 /= torch.norm(d2)

    losses = np.zeros((steps, steps))

    # Sample a few graphs to evaluate
    graphs = [generate_candidate_graph(random.randint(6, 20)) for _ in range(5)]

    for i, alpha in enumerate(np.linspace(-scale, scale, steps)):
        for j, beta in enumerate(np.linspace(-scale, scale, steps)):
            # New weights
            w_new = w0 + alpha * d1 + beta * d2
            set_flat_params(model, w_new)

            # Average loss over 5 graphs
            graph_losses = []
            for G in graphs:
                edge_index = list(G.edges())
                edge_features = get_contextual_edge_features(G, edge_index)
                scores = model(edge_features).detach().numpy()
                loss, _, _, _ = compute_loss(G, scores, edge_index, lambda_weight=0.5)
                graph_losses.append(loss)
            losses[i, j] = np.mean(graph_losses)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(losses, levels=50, cmap="viridis")
    plt.colorbar(label="Loss")
    plt.title("Loss Landscape around Trained Model")
    plt.xlabel("Direction α")
    plt.ylabel("Direction β")
    plt.tight_layout()
    plt.show()


#visualize_loss_surface("model33_best.pt", steps=30, scale=1.0)

def exploratory_restart(model_path, alpha_offset=1.2, beta_offset=1.2, save_as="model3_explored.pt", steps=30, scale=1.0):
    print(f"[Exploration] Loading model from: {model_path}")
    model = ContextAwareMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    w0 = get_flat_params(model)

    # Match direction generation from earlier
    torch.manual_seed(42)  # Optional: fix seed to get same directions
    d1 = torch.randn_like(w0)
    d2 = torch.randn_like(w0)
    d1 /= torch.norm(d1)
    d2 /= torch.norm(d2)

    # Display intended offset
    print(f"[Exploration] Jumping to α={alpha_offset:.3f}, β={beta_offset:.3f} in direction space")

    w_new = w0 + alpha_offset * d1 + beta_offset * d2
    set_flat_params(model, w_new)

    torch.save(model.state_dict(), save_as)
    print(f"[Exploration] New model saved to {save_as}")


def continue_training_from(model_path="model3_explored.pt", epochs=10, lambda_weight=0.5):
    model = ContextAwareMLP()
    model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    best_score = -1
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses, correct_trees = [], 0

        for _ in range(10):  # you can scale this up
            n_nodes = random.randint(6, 40)
            G = generate_candidate_graph(n_nodes)
            edge_index = list(G.edges())
            edge_features = get_contextual_edge_features(G, edge_index)
            scores = model(edge_features).detach().numpy()

            loss, is_correct, score, _ = compute_loss(G, scores, edge_index, lambda_weight)
            epoch_losses.append(loss)
            correct_trees += int(is_correct)

            # Backprop
            model.zero_grad()
            pred_scores = model(edge_features)
            pred_loss, _, _, _ = compute_loss(G, pred_scores.detach().numpy(), edge_index, lambda_weight)
            pred_loss = torch.tensor(pred_loss, requires_grad=True)
            pred_loss.backward()
            optimizer.step()

        avg_loss = np.mean(epoch_losses)
        print(f"[Exploratory] Epoch {epoch:02d} - Avg Loss: {avg_loss:.4f}, Correct Trees: {correct_trees}/10")

        if correct_trees > best_score:
            best_score = correct_trees
            torch.save(model.state_dict(), model_path)
            print(f"[Exploratory] Improved model saved to {model_path}")


exploratory_restart("model33_best.pt", alpha_offset=0.724, beta_offset=0.517)
continue_training_from("model3_explored.pt", epochs=10, lambda_weight=0.5)
