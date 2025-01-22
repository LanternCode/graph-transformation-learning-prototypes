in_channels = 32  # Node features dimensionality - currently learned in both models
hidden_channels = 128
out_channels = 2

epochs = 400
num_graphs = 1000  # The number of synthetic graphs to generate
num_nodes = 3000

learning_rate = 0.01
missing_edge_fraction = 0.2
edge_reconstruction_threshold = 0.8  # Construct an edge if the decoded probability is over the threshold
new_model_name = "gcn_bce_avg" # The name the trained model will be saved under

