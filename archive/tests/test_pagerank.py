import pytest
import numpy as np
import torch
import networkx as nx
from scipy import sparse

from src.algorithms.pagerank import MatrixPageRank
from src.traditional.pagerank import NetworkXPageRank

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

def test_matrix_pagerank_cpu():
    # Test PageRank on CPU
    adj_matrix = generate_test_graph()
    
    # Set parameters
    damping = 0.85
    max_iter = 100
    tol = 1e-6
    
    # Run matrix PageRank
    pr = MatrixPageRank(device='cpu', damping=damping, max_iter=max_iter, tol=tol)
    pr.preprocess(adj_matrix)
    scores = pr.run(adj_matrix.shape[0])
    
    # Run NetworkX PageRank for comparison
    nx_pr = NetworkXPageRank(damping=damping, max_iter=max_iter, tol=tol)
    nx_pr.preprocess(adj_matrix)
    nx_scores = nx_pr.run()
    
    # Convert NetworkX results to tensor
    nx_scores_tensor = torch.zeros(adj_matrix.shape[0])
    for node, score in nx_scores.items():
        nx_scores_tensor[node] = score
    
    # Compare results (allowing for some numerical differences)
    assert torch.allclose(scores, nx_scores_tensor, rtol=1e-4, atol=1e-4), "PageRank scores don't match"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matrix_pagerank_gpu():
    # Test PageRank on GPU (skip if no GPU available)
    adj_matrix = generate_test_graph()
    
    # Set parameters
    damping = 0.85
    max_iter = 100
    tol = 1e-6
    
    # Run matrix PageRank
    pr = MatrixPageRank(device='cuda', damping=damping, max_iter=max_iter, tol=tol)
    pr.preprocess(adj_matrix)
    scores = pr.run(adj_matrix.shape[0])
    
    # Run NetworkX PageRank for comparison
    nx_pr = NetworkXPageRank(damping=damping, max_iter=max_iter, tol=tol)
    nx_pr.preprocess(adj_matrix)
    nx_scores = nx_pr.run()
    
    # Convert NetworkX results to tensor
    nx_scores_tensor = torch.zeros(adj_matrix.shape[0], device='cuda')
    for node, score in nx_scores.items():
        nx_scores_tensor[node] = score
    
    # Compare results (allowing for some numerical differences)
    assert torch.allclose(scores, nx_scores_tensor, rtol=1e-4, atol=1e-4), "PageRank scores don't match"

def test_pagerank_convergence():
    # Test PageRank convergence on a directed graph
    adj_matrix = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ])
    adj_matrix = sparse.csr_matrix(adj_matrix)
    
    # Run matrix PageRank
    pr = MatrixPageRank(device='cpu', max_iter=1000)
    pr.preprocess(adj_matrix)
    scores = pr.run(adj_matrix.shape[0])
    
    # In this circular graph, all nodes should have the same PageRank
    expected_score = 1.0 / adj_matrix.shape[0]
    assert torch.allclose(scores, torch.tensor([expected_score] * adj_matrix.shape[0]), rtol=1e-4), "PageRank scores should be uniform"