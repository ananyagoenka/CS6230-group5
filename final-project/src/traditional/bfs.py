import networkx as nx
import numpy as np
from scipy import sparse

class NetworkXBFS:
    """
    Traditional implementation of BFS using NetworkX.
    """
    
    def __init__(self):
        """
        Initialize BFS algorithm.
        """
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
        
    def run(self, source):
        """
        Run BFS from source node.
        
        Args:
            source (int): Source node
            
        Returns:
            dict: Dictionary of distances
            dict: Dictionary of predecessors
        """
        if self.graph is None:
            raise ValueError("Call preprocess() first with the adjacency matrix")
        
        # Run BFS
        distances = {}
        predecessors = {}
        
        for node in self.graph:
            distances[node] = float('inf')
            predecessors[node] = -1
            
        distances[source] = 0
        
        # Using NetworkX's BFS implementation
        for node, (dist, pred) in nx.single_source_shortest_path_length(self.graph, source):
            distances[node] = dist
            if pred is not None:
                predecessors[node] = pred
                
        return distances, predecessors