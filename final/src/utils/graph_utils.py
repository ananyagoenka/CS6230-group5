import numpy as np
import torch
import networkx as nx
import scipy.sparse as sp
from time import time

def generate_random_graph(n, p, seed=None):
    """
    Generate a random Erdos-Renyi graph
    
    Parameters:
    -----------
    n : int
        Number of nodes
    p : float
        Probability of edge creation (between 0 and 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    G : networkx.Graph
        Generated random graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G

def generate_scale_free_graph(n, m, seed=None):
    """
    Generate a scale-free graph using Barabasi-Albert model
    
    Parameters:
    -----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    G : networkx.Graph
        Generated scale-free graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G

def generate_small_world_graph(n, k, p, seed=None):
    """
    Generate a small-world graph using Watts-Strogatz model
    
    Parameters:
    -----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    G : networkx.Graph
        Generated small-world graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G

def load_graph_from_edgelist(file_path, directed=False):
    """
    Load a graph from an edge list file
    
    Parameters:
    -----------
    file_path : str
        Path to the edge list file
    directed : bool
        Whether the graph is directed (default: False)
        
    Returns:
    --------
    G : networkx.Graph or networkx.DiGraph
        Loaded graph
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            # Parse edge
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                
                # Add weight if available
                if len(parts) >= 3:
                    weight = float(parts[2])
                    G.add_edge(u, v, weight=weight)
                else:
                    G.add_edge(u, v)
    
    return G

def graph_to_adj_list(G):
    """
    Convert a NetworkX graph to adjacency list representation
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph
        
    Returns:
    --------
    adj_list : dict
        Adjacency list representation
    """
    adj_list = {node: list(neighbors) for node, neighbors in G.adjacency()}
    return adj_list

def graph_to_adj_matrix_numpy(G):
    """
    Convert a NetworkX graph to adjacency matrix representation (NumPy)
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph
        
    Returns:
    --------
    adj_matrix : numpy.ndarray
        Adjacency matrix representation
    """
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix

def graph_to_adj_matrix_torch(G, device='cuda'):
    """
    Convert a NetworkX graph to adjacency matrix representation (PyTorch)
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph
    device : str
        Device to store the tensor (default: 'cuda')
        
    Returns:
    --------
    adj_matrix : torch.Tensor
        Adjacency matrix representation on the specified device
    """
    adj_numpy = nx.to_numpy_array(G)
    adj_tensor = torch.tensor(adj_numpy, dtype=torch.float32, device=device)
    return adj_tensor

def graph_to_sparse_adj_matrix_torch(G, device='cuda'):
    """
    Convert a NetworkX graph to sparse adjacency matrix representation (PyTorch)
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph
    device : str
        Device to store the tensor (default: 'cuda')
        
    Returns:
    --------
    adj_matrix : torch.sparse.Tensor
        Sparse adjacency matrix representation on the specified device
    """
    # Get sparse adjacency matrix in COO format
    adj_sparse = nx.to_scipy_sparse_array(G, format='coo')
    
    # Extract indices and values
    indices = torch.tensor(np.vstack((adj_sparse.row, adj_sparse.col)), 
                          dtype=torch.long, device=device)
    values = torch.tensor(adj_sparse.data, dtype=torch.float32, device=device)
    size = torch.Size(adj_sparse.shape)
    
    # Create sparse tensor
    adj_sparse_tensor = torch.sparse.FloatTensor(indices, values, size).to(device)
    
    return adj_sparse_tensor

def save_graph(G, file_path):
    """
    Save a NetworkX graph to a file using edge list format
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Graph to save
    file_path : str
        Path to save the graph
    """
    nx.write_edgelist(G, file_path, data=['weight'])

def print_graph_stats(G):
    """
    Print statistics about a graph
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph
    """
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    if nx.is_connected(G):
        print(f"Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
        print(f"Diameter: {nx.diameter(G)}")
    else:
        print("Graph is not connected")
    
    print(f"Density: {nx.density(G):.6f}")
    print(f"Clustering coefficient: {nx.average_clustering(G):.6f}")