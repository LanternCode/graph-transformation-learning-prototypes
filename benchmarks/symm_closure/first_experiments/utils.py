# Collate a dataset batch to GPU when processing it
import torch
from torch_geometric.data import Batch


def collate_to_device(data_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = Batch.from_data_list(data_list)
    return batch.to(device)  # Move batch to GPU


def prepare_edge_indices(batch):
    """Prepare positive and negative edge indices using explicitly provided training edges."""

    pos_edge_indices = []
    neg_edge_indices = []

    # Iterate through unique graphs in batch
    for graph_id in torch.unique(batch.batch):
        # Get mask for current graph
        mask = batch.batch == graph_id
        node_idx = torch.where(mask)[0]

        # Filter edges belonging to this graph
        edge_mask = mask[batch.train_edge_index[0]] & mask[batch.train_edge_index[1]]
        local_edges = batch.train_edge_index[:, edge_mask]
        local_labels = batch.train_edge_labels[edge_mask]

        # Map global indices to local indices
        local_mapping = {global_id.item(): local_id for local_id, global_id in enumerate(node_idx)}
        local_edges = torch.stack(
            [torch.tensor([local_mapping[e.item()] for e in edges_row]) for edges_row in local_edges])

        # Separate positive and negative edges
        pos_indices = local_edges[:, local_labels.to(local_edges.device) == 1]
        neg_indices = local_edges[:, local_labels.to(local_edges.device) == 0]

        pos_edge_indices.append(pos_indices)
        neg_edge_indices.append(neg_indices)

    return pos_edge_indices, neg_edge_indices
