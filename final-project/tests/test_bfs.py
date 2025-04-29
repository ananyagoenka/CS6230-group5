import pytest
import numpy as np
import torch
import networkx as nx
from scipy import sparse

from src.algorithms.bfs import MatrixBFS
from src.traditional.bfs import NetworkXBFS
from src.utils.sparse_utils import to_sparse_tensor

def generate_test_graph():
    # Create a simple test graph
    adj_matrix = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    return sparse.csr_matrix(adj_matrix)

def test_matrix_bfs_cpu():
    # Test BFS on CPU
    adj_matrix = generate_test_graph()
    
    # Run matrix BFS
    bfs = MatrixBFS(device='cpu')
    bfs.preprocess(adj_matrix)
    distances, predecessors = bfs.run(0, adj_matrix.shape[0])
    
    # Run NetworkX BFS for comparison
    nx_bfs = NetworkXBFS()
    nx_bfs.preprocess(adj_matrix)
    nx_distances, nx_predecessors = nx_bfs.run(0)
    
    # Convert NetworkX results to tensor
    nx_dist_tensor = torch.full((adj_matrix.shape[0],), float('inf'))
    for node, dist in nx_distances.items():
        nx_dist_tensor[node] = dist
    
    # Compare results
    assert torch.allclose(distances, nx_dist_tensor, rtol=1e-5), "BFS distances don't match"
    
    # Don't compare predecessors as they might be different but still valid

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matrix_bfs_gpu():
    # Test BFS on GPU (skip if no GPU available)
    adj_matrix = generate_test_graph()
    
    # Run matrix BFS
    bfs = MatrixBFS(device='cuda')
    bfs.preprocess(adj_matrix)
    distances, predecessors = bfs.run(0, adj_matrix.shape[0])
    
    # Run NetworkX BFS for comparison
    nx_bfs = NetworkXBFS()
    nx_bfs.preprocess(adj_matrix)
    nx_distances, nx_predecessors = nx_bfs.run(0)
    
    # Convert NetworkX results to tensor
    nx_dist_tensor = torch.full((adj_matrix.shape[0],), float('inf'), device='cuda')
    for node, dist in nx_distances.items():
        nx_dist_tensor[node] = dist
    
    # Compare results
    assert torch.allclose(distances, nx_dist_tensor, rtol=1e-5), "BFS distances don't match"

def test_bfs_disconnected_graph():
    # Test BFS on a disconnected graph
    adj_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    adj_matrix = sparse.csr_matrix(adj_matrix)
    
    # Run matrix BFS
    bfs = MatrixBFS(device='cpu')
    bfs.preprocess(adj_matrix)
    distances, predecessors = bfs.run(0, adj_matrix.shape[0])
    
    # Check results
    assert torch.isfinite(distances[0]), "Source node should have finite distance"
    assert torch.isfinite(distances[1]), "Node 1 should be reachable from source"
    assert not torch.isfinite(distances[2]), "Node 2 should not be reachable from source"
    assert not torch.isfinite(distances[3]), "Node 3 should not be reachable from source"
    assert not torch.isfinite(distances[4]), "Node 4 should not be reachable from source"