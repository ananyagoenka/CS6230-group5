import torch
import numpy as np
from scipy import sparse
import cupy as cp
from torch_sparse import SparseTensor
from src.utils.sparse_utils import csr_to_cuda, to_sparse_tensor

class MatrixBFS:
    """
    Linear algebra implementation of Breadth-First Search.
    
    BFS can be implemented using matrix-vector multiplication where:
    - The adjacency matrix represents the graph
    - A frontier vector represents nodes to explore at each level
    - Matrix-vector multiplication propagates the frontier
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize BFS algorithm.
        
        Args:
            device (str): Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        self.csr_matrix = None
        self.sparse_tensor = None
        
    def preprocess(self, adj_matrix):
        """
        Preprocess the adjacency matrix for optimized computation.
        
        Args:
            adj_matrix: Adjacency matrix in CSR format or numpy array
        """
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = sparse.csr_matrix(adj_matrix)
            
        if self.device == 'cuda':
            self.csr_matrix = csr_to_cuda(adj_matrix)
            self.sparse_tensor = to_sparse_tensor(adj_matrix).to(self.device)
        else:
            self.csr_matrix = adj_matrix
            self.sparse_tensor = to_sparse_tensor(adj_matrix)
    
    def run(self, source, num_nodes):
        """
        Run BFS from source node using matrix operations.
        
        Args:
            source (int): Source node to start BFS from
            num_nodes (int): Total number of nodes in the graph
            
        Returns:
            distances (torch.Tensor): Distance from source to each node
            predecessors (torch.Tensor): Predecessor of each node in BFS tree
        """
        if self.sparse_tensor is None:
            raise ValueError("Call preprocess() first with the adjacency matrix")
        
        # Initialize distances and predecessors
        distances = torch.full((num_nodes,), float('inf'), device=self.device)
        distances[source] = 0
        predecessors = torch.full((num_nodes,), -1, device=self.device)
        
        # Initialize frontier (vector of nodes to explore)
        frontier = torch.zeros(num_nodes, device=self.device)
        frontier[source] = 1
        
        level = 0
        while frontier.sum() > 0:
            level += 1
            
            # Propagate frontier using sparse matrix-vector multiplication
            # frontier = A * frontier (where A is the adjacency matrix)
            if self.device == 'cuda':
                next_frontier = torch.sparse.mm(self.sparse_tensor, frontier.view(-1, 1)).view(-1)
            else:
                next_frontier = torch.from_numpy(
                    self.csr_matrix.dot(frontier.cpu().numpy())
                ).to(self.device)
            
            # Only keep nodes that haven't been visited yet
            mask = (distances == float('inf')) & (next_frontier > 0)
            
            # Update distances and predecessors for newly discovered nodes
            if mask.sum() > 0:
                distances[mask] = level
                
                # For each newly discovered node, find a predecessor
                new_nodes = torch.nonzero(mask).view(-1)
                for node in new_nodes:
                    # Get neighbors from the transposed adjacency matrix
                    # In a typical adjacency matrix A, A[i,j] means there's an edge from i to j
                    # So we need to find j where A[j,i] > 0 and frontier[j] > 0
                    neighbors = torch.nonzero(self.sparse_tensor[:, node].to_dense() & frontier.bool()).view(-1)
                    if neighbors.numel() > 0:
                        predecessors[node] = neighbors[0]
            
            # Update frontier for next iteration
            frontier = torch.zeros_like(frontier)
            frontier[mask] = 1
        
        return distances, predecessors
        
    def run_batched(self, source, num_nodes, batch_size=1024):
        """
        Run BFS from source node using batched matrix operations for better memory efficiency.
        This is useful for very large graphs.
        
        Args:
            source (int): Source node to start BFS from
            num_nodes (int): Total number of nodes in the graph
            batch_size (int): Size of batches for matrix operations
            
        Returns:
            distances (torch.Tensor): Distance from source to each node
            predecessors (torch.Tensor): Predecessor of each node in BFS tree
        """
        # Similar implementation to run() but with batching
        # This is needed for very large graphs to avoid OOM errors
        # Implementation details omitted for brevity but would follow similar logic
        # with batched matrix operations
        pass