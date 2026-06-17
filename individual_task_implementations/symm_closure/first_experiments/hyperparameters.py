"""
Early feasibility prototype for symmetric-closure learning.

This file is part of the first round of exploratory experiments used to study
graph neural networks, graph autoencoders, adjacency-matrix learning, and basic
feasibility of reconstructing missing symmetric-closure edges. It is preserved
for historical context and reproducibility of the research process, not as a
clean final benchmark implementation. Later task-specific files supersede this
prototype for reported results.
"""
in_channels = 32  # Node features dimensionality - currently learned in both models
hidden_channels = 128
out_channels = 2

epochs = 200
num_graphs = 80  # The number of synthetic graphs to generate
num_nodes = 3000  # The max number of nodes in each synthetic graphs (min. 30)

learning_rate = 0.01  # When working with small datasets try 0.005 or 0.001
missing_edge_fraction = 0.2
edge_reconstruction_threshold = 0.8  # Construct an edge if the decoded probability is over the threshold
new_model_name = "gin_batch_concat"  # The name the trained model will be saved under
