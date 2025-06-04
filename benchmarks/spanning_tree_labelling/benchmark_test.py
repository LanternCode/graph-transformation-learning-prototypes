# Generate a smaller test batch to avoid memory issues
import random
from benchmark import generate_spanning_tree_graphs, evaluate_model


# Dummy model that randomly assigns 0 or 1 to each edge
def dummy_model_predict(graph):
    num_edges = len(graph['edge_index'])
    predicted_labels = [random.randint(0, 1) for _ in range(num_edges)]
    return predicted_labels


sample_graphs = generate_spanning_tree_graphs(num_graphs=1000)
evaluate_model(sample_graphs, dummy_model_predict)
