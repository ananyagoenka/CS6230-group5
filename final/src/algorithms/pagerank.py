import numpy as np
import torch
import time
from collections import defaultdict
from scipy.sparse import csr_matrix, diags
import multiprocessing as mp

# Import OpenMP Python bindings
try:
    import pymp
    HAS_PYMP = True
except ImportError:
    print("WARNING: pymp-pypi not found. Install with 'pip install pymp-pypi' for OpenMP support.")
    HAS_PYMP = False

class PageRank:
    """
    Class containing both traditional and linear algebra implementations of PageRank
    """
    
    @staticmethod
    def traditional_pagerank_cpu(adj_list, damping=0.85, max_iterations=100, tol=1e-6):
        """
        Traditional PageRank implementation using adjacency list
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
            
        Returns:
        --------
        ranks : dict
            Dictionary of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        n = len(adj_list)
        
        # Initialize PageRank scores
        ranks = {node: 1.0 / n for node in adj_list}
        
        # Calculate outgoing link counts
        outgoing_counts = {node: len(neighbors) for node, neighbors in adj_list.items()}
        
        # PageRank algorithm
        for iteration in range(max_iterations):
            new_ranks = {node: (1 - damping) / n for node in adj_list}
            
            # Update PageRank scores
            for node, neighbors in adj_list.items():
                for neighbor in neighbors:
                    if outgoing_counts[node] > 0:  # Avoid division by zero
                        new_ranks[neighbor] += damping * ranks[node] / outgoing_counts[node]
            
            # Check for convergence
            diff = sum(abs(new_ranks[node] - ranks[node]) for node in adj_list)
            ranks = new_ranks
            
            if diff < tol:
                return ranks, iteration + 1
        
        return ranks, max_iterations
    
    @staticmethod
    def traditional_pagerank_openmp(adj_list, damping=0.85, max_iterations=100, tol=1e-6, num_threads=None):
        """
        OpenMP-accelerated PageRank implementation using adjacency list
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
        num_threads : int or None
            Number of threads to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        ranks : dict
            Dictionary of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        if not HAS_PYMP:
            print("OpenMP not available, falling back to traditional PageRank")
            return PageRank.traditional_pagerank_cpu(adj_list, damping, max_iterations, tol)
            
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        n = len(adj_list)
        nodes = list(adj_list.keys())
        
        # Initialize PageRank scores
        ranks = {node: 1.0 / n for node in nodes}
        
        # Calculate outgoing link counts
        outgoing_counts = {node: len(neighbors) for node, neighbors in adj_list.items()}
        
        # PageRank algorithm with shared arrays
        for iteration in range(max_iterations):
            # Create new ranks for this iteration (no need for shared array)
            new_ranks = {node: (1 - damping) / n for node in nodes}
            
            # Process nodes in parallel
            with pymp.Parallel(num_threads) as p:
                # Local contribution dictionary for each thread
                local_contributions = [{} for _ in range(num_threads)]
                
                # Each thread processes a subset of nodes
                for i in p.range(n):
                    node = nodes[i]
                    thread_id = p.thread_num
                    
                    # Skip nodes with no outgoing links
                    if outgoing_counts[node] == 0:
                        continue
                    
                    # Calculate contribution to neighbors
                    contribution = damping * ranks[node] / outgoing_counts[node]
                    
                    # Update contributions locally
                    for neighbor in adj_list[node]:
                        if neighbor not in local_contributions[thread_id]:
                            local_contributions[thread_id][neighbor] = 0
                        local_contributions[thread_id][neighbor] += contribution
                
                # Combine local contributions with lock to avoid race conditions
                with p.lock:
                    for thread_contrib in local_contributions:
                        for node, contrib in thread_contrib.items():
                            new_ranks[node] += contrib
            
            # Check for convergence
            diff = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
            ranks = new_ranks
            
            if diff < tol:
                return ranks, iteration + 1
        
        return ranks, max_iterations
    
    @staticmethod
    def la_pagerank_cpu(adj_matrix, damping=0.85, max_iterations=100, tol=1e-6):
        """
        Linear algebra PageRank implementation using adjacency matrix on CPU
        
        Parameters:
        -----------
        adj_matrix : numpy.ndarray
            Adjacency matrix representation of the graph
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
            
        Returns:
        --------
        ranks : numpy.ndarray
            Array of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        n = adj_matrix.shape[0]
        
        # Calculate degree matrix (sum of outgoing links)
        out_degrees = np.sum(adj_matrix, axis=1)
        
        # Create transition matrix (M = A * D^-1)
        # Handle nodes with no outgoing links (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_out_degrees = np.where(out_degrees != 0, 1.0 / out_degrees, 0)
        
        # Convert to diagonal matrix for matrix multiplication
        inv_degree_matrix = np.diag(inv_out_degrees)
        transition_matrix = np.dot(adj_matrix.T, inv_degree_matrix)
        
        # Initialize PageRank vector
        ranks = np.ones(n) / n
        
        # Create teleportation vector
        teleport = np.ones(n) / n
        
        # PageRank algorithm
        for iteration in range(max_iterations):
            # Calculate new ranks
            new_ranks = damping * np.dot(transition_matrix, ranks) + (1 - damping) * teleport
            
            # Check for convergence
            if np.linalg.norm(new_ranks - ranks, 1) < tol:
                return new_ranks, iteration + 1
            
            ranks = new_ranks
        
        return ranks, max_iterations
    
    @staticmethod
    def la_pagerank_gpu(adj_matrix, damping=0.85, max_iterations=100, tol=1e-6):
        """
        Linear algebra PageRank implementation using adjacency matrix on GPU
        
        Parameters:
        -----------
        adj_matrix : torch.Tensor
            Adjacency matrix representation of the graph on GPU
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
            
        Returns:
        --------
        ranks : torch.Tensor
            Tensor of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        n = adj_matrix.shape[0]
        
        # Calculate degree matrix (sum of outgoing links)
        out_degrees = torch.sum(adj_matrix, dim=1)
        
        # Create transition matrix (M = A * D^-1)
        # Handle nodes with no outgoing links (avoid division by zero)
        inv_out_degrees = torch.where(out_degrees != 0, 1.0 / out_degrees, torch.zeros_like(out_degrees))
        
        # Convert to diagonal matrix for matrix multiplication
        inv_degree_matrix = torch.diag(inv_out_degrees)
        transition_matrix = torch.matmul(adj_matrix.t(), inv_degree_matrix)
        
        # Initialize PageRank vector
        ranks = torch.ones(n, device=adj_matrix.device) / n
        
        # Create teleportation vector
        teleport = torch.ones(n, device=adj_matrix.device) / n
        
        # PageRank algorithm
        for iteration in range(max_iterations):
            # Calculate new ranks
            new_ranks = damping * torch.matmul(transition_matrix, ranks) + (1 - damping) * teleport
            
            # Check for convergence
            if torch.norm(new_ranks - ranks, 1) < tol:
                return new_ranks, iteration + 1
            
            ranks = new_ranks
        
        return ranks, max_iterations
    
    @staticmethod
    def la_pagerank_sparse_gpu(adj_matrix, damping=0.85, max_iterations=100, tol=1e-6):
        """
        Linear algebra PageRank implementation using sparse adjacency matrix on GPU
        This is optimized for large sparse graphs
        
        Parameters:
        -----------
        adj_matrix : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph on GPU
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
            
        Returns:
        --------
        ranks : torch.Tensor
            Tensor of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        n = adj_matrix.shape[0]
        
        # Calculate out-degrees (sum of outgoing links)
        # For sparse matrix, we need to sum along columns (dim=1)
        out_degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        
        # Calculate transition matrix (M = A^T * D^-1)
        # First, create inverse out-degree vector (avoid division by zero)
        inv_out_degrees = torch.where(out_degrees != 0, 1.0 / out_degrees, torch.zeros_like(out_degrees))
        
        # For sparse implementation, we don't explicitly create the transition matrix
        # Instead, we perform the multiplication in steps during each iteration
        
        # Initialize PageRank vector
        ranks = torch.ones(n, device=adj_matrix.device) / n
        
        # Create teleportation vector
        teleport = torch.ones(n, device=adj_matrix.device) / n
        
        # Create transposed adjacency matrix for more efficient calculations
        adj_matrix_t = adj_matrix.transpose(0, 1)
        
        # PageRank algorithm
        for iteration in range(max_iterations):
            # Step 1: Multiply current ranks by inverse out-degrees
            scaled_ranks = ranks * inv_out_degrees
            
            # Step 2: Multiply by transpose of adjacency matrix (sparse multiplication)
            new_ranks = damping * torch.sparse.mm(adj_matrix_t, scaled_ranks.unsqueeze(1)).squeeze(1)
            
            # Step 3: Add teleportation component
            new_ranks += (1 - damping) * teleport
            
            # Check for convergence
            if torch.norm(new_ranks - ranks, 1) < tol:
                return new_ranks, iteration + 1
            
            ranks = new_ranks
        
        return ranks, max_iterations