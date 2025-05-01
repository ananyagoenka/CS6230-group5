import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import os
from pathlib import Path

def read_edgelist(file_path, directed=False, weighted=False, delimiter=None):
    """
    Read graph from edge list file.
    
    Args:
        file_path (str): Path to edge list file
        directed (bool): Whether the graph is directed
        weighted (bool): Whether the graph has edge weights
        delimiter (str): Delimiter used in the file
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
        int: Number of nodes
    """
    # Read edges
    if weighted:
        edges = np.loadtxt(file_path, delimiter=delimiter, dtype=np.float32)
        src = edges[:, 0].astype(np.int32)
        dst = edges[:, 1].astype(np.int32)
        weights = edges[:, 2]
    else:
        edges = np.loadtxt(file_path, delimiter=delimiter, dtype=np.int32)
        src = edges[:, 0]
        dst = edges[:, 1]
        weights = np.ones(len(src))
    
    # Find number of nodes
    num_nodes = max(src.max(), dst.max()) + 1
    
    # Create sparse adjacency matrix
    adj = sp.csr_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))
    
    # Make symmetric if undirected
    if not directed:
        adj = (adj + adj.T) / 2
    
    return adj, num_nodes

def read_mtx(file_path):
    """
    Read graph from MatrixMarket file.
    
    Args:
        file_path (str): Path to MatrixMarket file
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
        int: Number of nodes
    """
    adj = sp.io.mmread(file_path).tocsr()
    return adj, adj.shape[0]

def save_sparse_matrix(file_path, adj):
    """
    Save sparse adjacency matrix to file.
    
    Args:
        file_path (str): Path to save the matrix
        adj: Sparse adjacency matrix
    """
    if isinstance(adj, torch.Tensor):
        # Convert PyTorch tensor to scipy sparse
        if adj.is_sparse:
            indices = adj._indices().cpu().numpy()
            values = adj._values().cpu().numpy()
            shape = adj.size()
            adj = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
        else:
            adj = sp.csr_matrix(adj.cpu().numpy())
    
    # Save in different formats based on extension
    ext = os.path.splitext(file_path)[1]
    if ext == '.npz':
        sp.save_npz(file_path, adj)
    elif ext == '.mtx':
        sp.io.mmwrite(file_path, adj)
    else:
        # Default to NPZ
        sp.save_npz(file_path, adj)

def load_sparse_matrix(file_path):
    """
    Load sparse adjacency matrix from file.
    
    Args:
        file_path (str): Path to the matrix file
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    ext = os.path.splitext(file_path)[1]
    if ext == '.npz':
        return sp.load_npz(file_path)
    elif ext == '.mtx':
        return sp.io.mmread(file_path).tocsr()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def networkx_to_sparse(G):
    """
    Convert NetworkX graph to sparse adjacency matrix.
    
    Args:
        G: NetworkX graph
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
        int: Number of nodes
    """
    return nx.to_scipy_sparse_matrix(G), G.number_of_nodes()

def sparse_to_networkx(adj, directed=False):
    """
    Convert sparse adjacency matrix to NetworkX graph.
    
    Args:
        adj: Sparse adjacency matrix
        directed (bool): Whether to create a directed graph
        
    Returns:
        networkx.Graph or networkx.DiGraph: NetworkX graph
    """
    if directed:
        G = nx.DiGraph(adj)
    else:
        G = nx.Graph(adj)
    return G