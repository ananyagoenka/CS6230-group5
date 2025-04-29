import pytest
import numpy as np
import torch
import networkx as nx
from scipy import sparse

from src.algorithms.connected_components import MatrixConnectedComponents
from src.traditional.connected_components import NetworkXCC

def generate_test_graph():
    # Create a simple test graph with 3 connected components
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])
    return sparse.csr_matrix(adj_matrix)

def test_matrix_cc_cpu():
    # Test Connected Components on CPU
    adj_matrix = generate_test_graph()
    
    # Run matrix Connected Components
    cc = MatrixConnectedComponents(device='cpu')
    cc.preprocess(adj_matrix)
    components = cc.run(adj_matrix.shape[0])
    
    # Run NetworkX Connected Components for comparison
    nx_cc = NetworkXCC()
    nx_cc.preprocess(adj_matrix)
    nx_components = nx_cc.run()
    
    # Convert NetworkX results to tensor
    nx_comp_tensor = torch.zeros(adj_matrix.shape[0], dtype=torch.long)
    for node, comp in nx_components.items():
        nx_comp_tensor[node] = comp
    
    # Count number of unique components
    num_components = len(torch.unique(components))
    nx_num_components = len(torch.unique(nx_comp_tensor))
    
    # Compare results
    assert num_components == nx_num_components, "Number of components doesn't match"
    
    # Check connected nodes have the same component
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if adj_matrix[i, j] != 0:
                assert components[i] == components[j], f"Nodes {i} and {j} should be in the same component"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matrix_cc_gpu():
    # Test Connected Components on GPU (skip if no GPU available)
    adj_matrix = generate_test_graph()
    
    # Run matrix Connected Components
    cc = MatrixConnectedComponents(device='cuda')
    cc.preprocess(adj_matrix)
    components = cc.run(adj_matrix.shape[0])
    
    # Run NetworkX Connected Components for comparison
    nx_cc = NetworkXCC()
    nx_cc.preprocess(adj_matrix)
    nx_components = nx_cc.run()
    
    # Count number of unique components
    num_components = len(torch.unique(components))
    nx_num_components = len(set(nx_components.values()))
    
    # Compare results
    assert num_components == nx_num_components, "Number of components doesn't match"

def test_cc_single_component():
    # Test with a single connected component
    adj_matrix = np.ones((5, 5)) - np.eye(5)
    adj_matrix = sparse.csr_matrix(adj_matrix)
    
    # Run matrix Connected Components
    cc = MatrixConnectedComponents(device='cpu')
    cc.preprocess(adj_matrix)
    components = cc.run(adj_matrix.shape[0])
    
    # All nodes should have the same component
    assert len(torch.unique(components)) == 1, "Should have only one component"