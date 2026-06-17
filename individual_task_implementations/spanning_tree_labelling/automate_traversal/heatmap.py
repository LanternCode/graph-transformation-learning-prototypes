"""
Failed local parameter-space search experiment.

This script was an exploratory attempt to move a trained context-aware model
through random two-dimensional slices of parameter space, visualize the local
loss/correctness landscape, and identify nearby checkpoints with better
spanning-tree reconstruction behavior. It is preserved as a research artifact
showing an attempted automated search strategy, not as a validated
hyperparameter-optimization method or final model-selection pipeline.
The code may not compile or work as expected.
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualise_and_retrain import get_flat_params, set_flat_params
from test_three import ContextAwareMLP, generate_candidate_graph, get_contextual_edge_features, compute_loss, \
    evaluate_model


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#seed_everything()


def grid_search_correctness_with_save(model_path="model3_best.pt", steps=30, scale=1.0):
    print(f"[Grid Search] Loading model from: {model_path}")
    model = ContextAwareMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    w0 = get_flat_params(model)
    torch.manual_seed(42)
    d1 = torch.randn_like(w0)
    d2 = torch.randn_like(w0)
    d1 /= torch.norm(d1)
    d2 /= torch.norm(d2)

    losses = np.zeros((steps, steps))
    corrects = np.zeros((steps, steps))

    graphs = [generate_candidate_graph(random.randint(6, 20)) for _ in range(5)]

    best_score = -1
    best_coords = (0, 0)

    for i, alpha in enumerate(np.linspace(-scale, scale, steps)):
        for j, beta in enumerate(np.linspace(-scale, scale, steps)):
            w_new = w0 + alpha * d1 + beta * d2
            set_flat_params(model, w_new)

            total_loss = 0
            total_correct = 0

            for G in graphs:
                edge_index = list(G.edges())
                edge_features = get_contextual_edge_features(G, edge_index)
                scores = model(edge_features).detach().numpy()
                loss, is_correct, _, _ = compute_loss(G, scores, edge_index, lambda_weight=0.5)
                total_loss += loss
                total_correct += is_correct

            losses[i, j] = total_loss / len(graphs)
            corrects[i, j] = total_correct

            if total_correct > best_score:
                best_score = total_correct
                best_coords = (alpha, beta)

    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axs[0].imshow(losses, cmap='viridis', origin='lower')
    axs[0].set_title("Average Loss")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(corrects, cmap='Blues', origin='lower')
    axs[1].set_title("Correct Trees")
    plt.colorbar(im1, ax=axs[1])

    for ax in axs:
        ax.set_xlabel("β offset")
        ax.set_ylabel("α offset")

    plt.suptitle("Model Landscape: Loss & Correct Trees")
    plt.tight_layout()
    plt.show()

    # === Prompt to save ===
    user_input = input(f"\nBest correct trees = {best_score}/5 at α={best_coords[0]:.3f}, β={best_coords[1]:.3f}.\nSave this model? (y/n): ").strip().lower()
    if user_input == "y":
        model = ContextAwareMLP()
        model.load_state_dict(torch.load(model_path))
        w0 = get_flat_params(model)
        w_new = w0 + best_coords[0] * d1 + best_coords[1] * d2
        set_flat_params(model, w_new)
        torch.save(model.state_dict(), "model3_selected_from_grid.pt")
        print("[Grid Search] Model saved as model3_selected_from_grid.pt")
    else:
        print("[Grid Search] Model not saved.")


def automated_grid_search_until_hit(model_path="model3_best.pt", steps=30, scale=1.0, min_correct=3, max_attempts=50):
    print(f"[Auto Search] Searching for a region with ≥ {min_correct}/10 correct trees...")
    base_model = ContextAwareMLP()
    base_model.load_state_dict(torch.load(model_path))
    w0 = get_flat_params(base_model)

    for attempt in range(1, max_attempts + 1):
        print(f"\n[Auto Search] Attempt {attempt}/{max_attempts}")
        model = ContextAwareMLP()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Random directions (fresh seed each attempt)
        d1 = torch.randn_like(w0)
        d2 = torch.randn_like(w0)
        d1 /= torch.norm(d1)
        d2 /= torch.norm(d2)

        corrects = np.zeros((steps, steps))
        graphs = [generate_candidate_graph(random.randint(6, 20)) for _ in range(1000)]

        best_score = -1
        best_coords = (0, 0)

        for i, alpha in enumerate(np.linspace(-scale, scale, steps)):
            for j, beta in enumerate(np.linspace(-scale, scale, steps)):
                w_new = w0 + alpha * d1 + beta * d2
                set_flat_params(model, w_new)

                total_correct = 0
                for G in graphs:
                    edge_index = list(G.edges())
                    edge_features = get_contextual_edge_features(G, edge_index)
                    scores = model(edge_features).detach().numpy()
                    _, is_correct, _, _ = compute_loss(G, scores, edge_index, lambda_weight=0.5)
                    total_correct += is_correct

                corrects[i, j] = total_correct
                if total_correct > best_score:
                    best_score = total_correct
                    best_coords = (alpha, beta)

        print(f"[Auto Search] Best found: {best_score}/10 correct at α={best_coords[0]:.3f}, β={best_coords[1]:.3f}")

        if best_score >= min_correct:
            # Save that model
            final_model = ContextAwareMLP()
            final_model.load_state_dict(torch.load(model_path))
            w0 = get_flat_params(final_model)
            w_new = w0 + best_coords[0] * d1 + best_coords[1] * d2
            set_flat_params(final_model, w_new)
            torch.save(final_model.state_dict(), "model3_auto_selected.pt")
            print("[Auto Search] Success! Model saved as model3_auto_selected.pt")
            return best_coords

    print("[Auto Search] No region met the threshold after max attempts.")
    return None


#grid_search_correctness_with_save("model3_explored.pt", steps=30, scale=1.0)
automated_grid_search_until_hit("model33_best.pt", min_correct=100, steps=30, scale=1.0, max_attempts=100)
#evaluate_model("model3_auto_selected.pt")
