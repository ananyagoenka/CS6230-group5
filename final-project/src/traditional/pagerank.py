import networkx as nx
import numpy as np
from scipy import sparse

class NetworkXPageRank:
    """
    Traditional implementation of PageRank using NetworkX.
    """
    
    def __init__(self, damping=0.85, max_iter=100, tol=1e-6):
        """
        Initialize PageRank algorithm.
        
        Args:
            damping (float): Damping factor
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        """
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.graph = None
        
    def preprocess(self, adj_matrix):
        """
        Preprocess the adjacency matrix to create a NetworkX graph.
        
        Args:
            adj_matrix: Adjacency matrix in CSR format or numpy array
        """
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = sparse.csr_matrix(adj_matrix)
            
        # Convert to NetworkX graph
        self.graph = nx.from_scipy_sparse_matrix(adj_matrix, create_using=nx.DiGraph)
        
    def run(self):
        """
        Run PageRank algorithm.
            
        Returns:
            dict: PageRank scores for each node
        """
        if self.graph is None:
            raise ValueError("Call preprocess() first with the adjacency matrix")
        
        # Run PageRank
        return nx.pagerank(
            self.graph,
            alpha=self.damping,
            max_iter=self.max_iter,
            tol=self.tol
        )