import torch
import numpy as np
from scipy import sparse
import cupy as cp
from src.utils.sparse_utils import csr_to_cuda, to_sparse_tensor

class MatrixPageRank:
    """
    Linear algebra implementation of the PageRank algorithm.
    
    PageRank can be efficiently implemented as an iterative matrix-vector 
    multiplication until convergence.
    """
    
    def __init__(self, device='cuda', damping=0.85, max_iter=100, tol=1e-6):
        """
        Initialize PageRank algorithm.
        
        Args:
            device (str): Device to run computations on ('cuda' or 'cpu')
            damping (float): Damping factor (probability of following links)
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.device = device
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.transition_matrix = None
        
    def preprocess(self, adj_matrix):
        """
        Preprocess the adjacency matrix to create a column-stochastic transition matrix.
        
        Args:
            adj_matrix: Adjacency matrix in CSR format or numpy array
        """
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = sparse.csr_matrix(adj_matrix)
        
        # Convert adjacency matrix to transition matrix
        # First, compute out-degree of each node
        out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        
        # Create diagonal matrix with 1/out_degree
        with np.errstate(divide='ignore'):
            inv_out_degrees = np.divide(1.0, out_degrees)
        inv_out_degrees[np.isinf(inv_out_degrees)] = 0.0
        
        # Diagonal matrix for normalization
        D_inv = sparse.diags(inv_out_degrees)
        
        # Compute transition matrix (column-stochastic)
        transition = adj_matrix.T.dot(D_inv)
        
        if self.device == 'cuda':
            self.transition_matrix = to_sparse_tensor(transition).to(self.device)
        else:
            self.transition_matrix = to_sparse_tensor(transition)
            
    def run(self, num_nodes):
        """
        Run PageRank algorithm using matrix operations.
        
        Args:
            num_nodes (int): Total number of nodes in the graph
            
        Returns:
            scores (torch.Tensor): PageRank scores for each node
        """
        if self.transition_matrix is None:
            raise ValueError("Call preprocess() first with the adjacency matrix")
        
        # Initialize PageRank vector with uniform distribution
        scores = torch.full((num_nodes,), 1.0 / num_nodes, device=self.device)
        
        # Teleportation vector (uniform)
        teleport = torch.full((num_nodes,), 1.0 / num_nodes, device=self.device)
        
        # Power iteration method
        for i in range(self.max_iter):
            prev_scores = scores.clone()
            
            # PageRank iteration: r = d * M * r + (1-d) * v
            # where:
            # - r is the PageRank vector
            # - M is the transition matrix
            # - d is the damping factor
            # - v is the teleportation vector
            
            # Compute M * r (sparse matrix-vector multiplication)
            if self.device == 'cuda':
                mult = torch.sparse.mm(self.transition_matrix, scores.view(-1, 1)).view(-1)
            else:
                mult = torch.matmul(self.transition_matrix, scores.view(-1, 1)).view(-1)
            
            # Apply damping and teleportation
            scores = self.damping * mult + (1 - self.damping) * teleport
            
            # Check convergence
            diff = torch.norm(scores - prev_scores, p=1)
            if diff < self.tol:
                break
        
        # Normalize scores
        scores /= scores.sum()
        
        return scores
        
    def run_distributed(self, num_nodes, num_gpus=1):
        """
        Run PageRank with multi-GPU distribution.
        
        Args:
            num_nodes (int): Total number of nodes in the graph
            num_gpus (int): Number of GPUs to use
            
        Returns:
            scores (torch.Tensor): PageRank scores for each node
        """
        # Implementation for multi-GPU scaling
        # Involves splitting the transition matrix and PageRank vector across GPUs
        # Using collective operations for communication
        # Details omitted for brevity
        pass