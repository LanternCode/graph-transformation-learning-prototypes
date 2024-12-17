import argparse
import torch
import hyperparameters
from eval import decode_all
from gen_sym_closure import get_data
from model_gae_gcn import DirectedGAEGCN
from model_gae_gin import DirectedGAEGIN
from train import train_gae_gcn, train_gae_gin, train_small_gae_gcn


def main(args):
    if args.obj == "training":
        if args.model == "gcn":
            if args.lf == "avg_small":
                train_small_gae_gcn()
            else:
                train_gae_gcn(args.lf)
        elif args.model == "gin":
            train_gae_gin()
    elif args.obj == "inference":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model == "gcn":
            model = DirectedGAEGCN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                                   num_nodes=hyperparameters.num_nodes, device=device)
            if args.lf == "avg":
                model.load_state_dict(torch.load('trained_many_gae_gcn_alt_loss.pth'))
            elif args.lf == "bce":
                print("")
                # model.load_state_dict(torch.load('trained_many_gae_gcn_alt_loss.pth'))
            elif args.lf == "avg_small":
                model.load_state_dict(torch.load('gcn_small.pth'))
        elif args.model == "gin":
            model = DirectedGAEGIN(out_channels=hyperparameters.out_channels, hidden_channels=hyperparameters.hidden_channels,
                                   num_nodes=hyperparameters.num_nodes)
            model.load_state_dict(torch.load('trained_gae_gin.pth'))

        # Set the model to evaluation mode and perform regular or fully connected decoding
        num_inference_nodes = 1000
        inference_data = get_data(num_inference_nodes, hyperparameters.missing_edge_fraction)
        inference_data.batch = torch.zeros(inference_data.x.size(0), dtype=torch.long)  # Dummy batch
        model.eval()
        decode_all(model, inference_data, inference_data.x.size(0), 0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different versions of the model with custom configurations.")

    # Objective
    parser.add_argument('--obj', type=str, default='training',
                        choices=['training', 'inference'],
                        help="Training or inference")

    # Model
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gin'],
                        help="GCN or GIN")

    # Loss function
    parser.add_argument('--lf', type=str, default='bce',
                        choices=['bce', 'avg', 'avg_small'],
                        help="Loss function to use (bce-based with removed_edges, average-based with removed_edges, average-based without removed_edges).")

    # Parse arguments
    args = parser.parse_args()

    # Pass arguments to the main function
    main(args)