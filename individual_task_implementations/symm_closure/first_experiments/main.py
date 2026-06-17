"""
Early feasibility prototype for symmetric-closure learning.

This file is part of the first round of exploratory experiments used to study
graph neural networks, graph autoencoders, adjacency-matrix learning, and basic
feasibility of reconstructing missing symmetric-closure edges. It is preserved
for historical context and reproducibility of the research process, not as a
clean final benchmark implementation. Later task-specific files supersede this
prototype for reported results.
"""
import argparse
import torch
import hyperparameters
from eval import decode_all, decode_and_union_all, decode_all_with_directions
from gen_sym_closure import get_data
from model_gae_gcn import DirectedGAEGCN
from train_gae_gcn import train_gae_gcn, train_small_gae_gcn
from model_gae_gin import DirectedGAEGIN
from train_gae_gin import train_gae_gin


def main(args):
    if args.obj == "training":
        if args.model == "gcn":
            if args.lf == "small":
                train_small_gae_gcn()
            else:
                train_gae_gcn(args.lf)
        elif args.model == "gin":
            train_gae_gin()
        elif args.model == "kWL":
            train_small()
    elif args.obj == "inference":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == "gcn":
            model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                                   num_nodes=hyperparameters.num_nodes, device=device)
            if args.lf == "avg":
                model.load_state_dict(torch.load('results/trained_many_gae_gcn_alt_loss.pth'))
            elif args.lf == "bce":
                model.load_state_dict(torch.load('results/gcn_bce_avg.pth'))
            elif args.lf == "small":
                model.load_state_dict(torch.load('concat.pth'))
        elif args.model == "gin":
            model = DirectedGAEGIN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                                   num_nodes=hyperparameters.num_nodes, device=device)
            model.load_state_dict(torch.load('gin_batch_concat_full.pth'))

        # Set the model to evaluation mode and perform regular or fully connected decoding
        num_inference_nodes = 10
        model.eval()

        if args.lf == "small":
            # List of thresholds for decode_all on a tiny dataset
            thresholds = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
            for threshold in thresholds:
                inference_data = get_data(num_inference_nodes, hyperparameters.missing_edge_fraction)
                inference_data.batch = torch.zeros(inference_data.x.size(0), dtype=torch.long)  # Dummy batch
                decode_all_with_directions(model, inference_data, inference_data.num_nodes, threshold)
        else:
            # Inference on a full dataset
            thresholds = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
            #thresholds = [0.5, 0.45, 0.4, 0.35, 0.3]
            for threshold in thresholds:
                inference_data = get_data(num_inference_nodes, hyperparameters.missing_edge_fraction)
                inference_data.batch = torch.zeros(inference_data.x.size(0), dtype=torch.long)  # Dummy batch
                decode_all(model, inference_data, inference_data.x.size(0), threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different versions of the model with custom configurations.")

    # Objective
    parser.add_argument('--obj', type=str, default='training',
                        choices=['training', 'inference'],
                        help="Training or inference mode")

    # Model
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gin', 'kWL'],
                        help="GCN, GIN, or k-WL GNN")

    # Loss function
    parser.add_argument('--lf', type=str, default='bce',
                        choices=['bce', 'avg', 'small', 'default'],
                        help="Loss function to use (bce-based with removed_edges, average-based with removed_edges, small without removed_edges).")

    # Parse arguments
    args = parser.parse_args()

    # Pass arguments to the main function
    main(args)