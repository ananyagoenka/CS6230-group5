import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from pathlib import Path

def generate_erdos_renyi_graph(n, p, seed=None, directed=False):
    """
    Generate an Erdos-Renyi random graph.
    
    Args:
        n (int): Number of nodes
        p (float): Probability of edge creation
        seed (int): Random seed
        directed (bool): Whether to create a directed graph
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=directed)
    return nx.to_scipy_sparse_matrix(G)

def generate_barabasi_albert_graph(n, m, seed=None):
    """
    Generate a Barabasi-Albert preferential attachment graph.
    
    Args:
        n (int): Number of nodes
        m (int): Number of edges to attach from a new node to existing nodes
        seed (int): Random seed
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return nx.to_scipy_sparse_matrix(G)

def generate_watts_strogatz_graph(n, k, p, seed=None):
    """
    Generate a Watts-Strogatz small-world graph.
    
    Args:
        n (int): Number of nodes
        k (int): Each node is connected to k nearest neighbors in ring topology
        p (float): Probability of rewiring each edge
        seed (int): Random seed
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return nx.to_scipy_sparse_matrix(G)

def generate_sbm_graph(sizes, probs, seed=None):
    """
    Generate a Stochastic Block Model graph.
    
    Args:
        sizes (list): List of community sizes
        probs (list): Matrix of inter-community edge probabilities
        seed (int): Random seed
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return nx.to_scipy_sparse_matrix(G)

def generate_rmat_graph(scale, edge_factor, a=0.45, b=0.15, c=0.15, seed=None):
    """
    Generate an R-MAT graph (Recursive Matrix) - approximated using NetworkX.
    
    Args:
        scale (int): Log2 of number of vertices
        edge_factor (int): Ratio of edges to vertices
        a, b, c (float): R-MAT parameters
        seed (int): Random seed
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = 2**scale
    m = n * edge_factor
    
    # Use NetworkX scale-free directed graph as approximation
    # The actual R-MAT model would require a specialized implementation
    G = nx.scale_free_graph(n, alpha=a, beta=b, gamma=c, seed=seed)
    
    # Keep only the top m edges
    edges = list(G.edges())
    if len(edges) > m:
        np.random.shuffle(edges)
        edges = edges[:m]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
    
    return nx.to_scipy_sparse_matrix(G)

def save_graph(adj_matrix, filename, format='npz'):
    """
    Save a graph adjacency matrix to file.
    
    Args:
        adj_matrix: Adjacency matrix in scipy.sparse format
        filename (str): Output filename
        format (str): Output format ('npz', 'mtx', or 'edgelist')
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if format == 'npz':
        sp.save_npz(filename, adj_matrix)
    elif format == 'mtx':
        sp.io.mmwrite(filename, adj_matrix)
    elif format == 'edgelist':
        # Convert to edgelist
        rows, cols = adj_matrix.nonzero()
        data = adj_matrix.data
        with open(filename, 'w') as f:
            for i in range(len(rows)):
                f.write(f"{rows[i]} {cols[i]} {data[i]}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")

def generate_all_graphs():
    """
    Generate a suite of synthetic graphs with various properties.
    """
    output_dir = Path("generated")
    output_dir.mkdir(exist_ok=True)
    
    # Small graphs for testing
    print("Generating small graphs for testing...")
    save_graph(generate_erdos_renyi_graph(100, 0.05, seed=42), 
               output_dir / "er_small.npz")
    save_graph(generate_barabasi_albert_graph(100, 5, seed=42), 
               output_dir / "ba_small.npz")
    save_graph(generate_watts_strogatz_graph(100, 10, 0.1, seed=42), 
               output_dir / "ws_small.npz")
    
    # Medium graphs
    print("Generating medium graphs...")
    save_graph(generate_erdos_renyi_graph(10000, 0.001, seed=42), 
               output_dir / "er_medium.npz")
    save_graph(generate_barabasi_albert_graph(10000, 10, seed=42), 
               output_dir / "ba_medium.npz")
    
    # Large graphs
    print("Generating large graphs...")
    save_graph(generate_rmat_graph(20, 16, seed=42), 
               output_dir / "rmat_large.npz")
    
    # Community graphs
    print("Generating graphs with community structure...")
    sizes = [1000, 1000, 1000, 1000]
    probs = [[0.1, 0.01, 0.01, 0.01],
             [0.01, 0.1, 0.01, 0.01],
             [0.01, 0.01, 0.1, 0.01],
             [0.01, 0.01, 0.01, 0.1]]
    save_graph(generate_sbm_graph(sizes, probs, seed=42), 
               output_dir / "sbm_communities.npz")

if __name__ == "__main__":
    generate_all_graphs()
    print("All synthetic graphs generated successfully.")