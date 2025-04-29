import networkx as nx
import numpy as np
from scipy import sparse

class NetworkXCC:
    """
    Traditional implementation of Connected Components using NetworkX.
    """
    
    def __init__(self):
        """
        Initialize Connected Components algorithm.
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
            
        # Make sure the matrix is symmetric (undirected graph)
        adj_matrix = (adj_matrix + adj_matrix.T) > 0
            
        # Convert to NetworkX graph
        self.graph = nx.from_scipy_sparse_matrix(adj_matrix, create_using=nx.Graph)
        
    def run(self):
        """
        Run Connected Components algorithm.
            
        Returns:
            dict: Component label for each node
        """
        if self.graph is None:
            raise ValueError("Call preprocess() first with the adjacency matrix")
        
        # Run Connected Components
        components = nx.connected_components(self.graph)
        
        # Create a dictionary mapping node -> component ID
        component_labels = {}
        for i, component in enumerate(components):
            for node in component:
                component_labels[node] = i
                
        return component_labels