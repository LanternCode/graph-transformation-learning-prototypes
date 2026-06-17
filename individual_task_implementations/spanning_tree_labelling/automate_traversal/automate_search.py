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

from visualise_and_retrain import set_flat_params, get_flat_params
from test_three import ContextAwareMLP, generate_candidate_graph, get_contextual_edge_features, compute_loss, \
    evaluate_model

def automated_grid_search_until_hit(model_path="model3_best.pt", steps=30, scale=1.0, min_correct=300, max_attempts=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Auto Search] Using device: {device}")
    print(f"[Auto Search] Searching for a region with ≥ {min_correct}/1000 correct trees...")

    base_model = ContextAwareMLP().to(device)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    w0 = get_flat_params(base_model).to(device)

    for attempt in range(1, max_attempts + 1):
        print(f"\n[Auto Search] Attempt {attempt}/{max_attempts}")
        model = ContextAwareMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        d1 = torch.randn_like(w0, device=device)
        d2 = torch.randn_like(w0, device=device)
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
                    edge_features = get_contextual_edge_features(G, edge_index).to(device)
                    scores = model(edge_features).detach().cpu().numpy()
                    _, is_correct, _, _ = compute_loss(G, scores, edge_index, lambda_weight=0.5)
                    total_correct += is_correct

                corrects[i, j] = total_correct
                if total_correct > best_score:
                    best_score = total_correct
                    best_coords = (alpha, beta)

        print(f"[Auto Search] Best found: {best_score}/1000 correct at α={best_coords[0]:.3f}, β={best_coords[1]:.3f}")

        if best_score >= min_correct:
            final_model = ContextAwareMLP().to(device)
            final_model.load_state_dict(torch.load(model_path, map_location=device))
            w0 = get_flat_params(final_model).to(device)
            w_new = w0 + best_coords[0] * d1 + best_coords[1] * d2
            set_flat_params(final_model, w_new)
            torch.save(final_model.state_dict(), "model3_auto_selected.pt")
            print("[Auto Search] Success! Model saved as model3_auto_selected.pt")
            return best_coords

    print("[Auto Search] No region met the threshold after max attempts.")
    return None



#grid_search_correctness_with_save("model3_explored.pt", steps=30, scale=1.0)
automated_grid_search_until_hit("model33_best.pt", min_correct=1600, steps=30, scale=1.0, max_attempts=1000)
#evaluate_model("model3_auto_selected.pt")