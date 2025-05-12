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
        OpenMP-accelerated PageRank implementation
        
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
        pagerank : dict
            Dictionary mapping node IDs to PageRank scores
        iterations : int
            Number of iterations performed
        """
        if not HAS_PYMP:
            print("OpenMP not available, falling back to traditional PageRank")
            return PageRank.traditional_pagerank_cpu(adj_list, damping, max_iterations, tol)
            
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        # Get list of nodes
        nodes = list(adj_list.keys())
        n = len(nodes)
        
        # Calculate out-degrees
        out_degrees = {node: len(neighbors) for node, neighbors in adj_list.items()}
        
        # Initialize PageRank scores
        pagerank = {node: 1.0 / n for node in nodes}
        
        # Run PageRank algorithm
        converged = False
        iterations = 0
        
        while not converged and iterations < max_iterations:
            # Create a new rank dictionary for this iteration
            new_pagerank = {node: (1.0 - damping) / n for node in nodes}
            
            # Collect redistribution values sequentially to avoid race conditions
            with pymp.Parallel(num_threads) as p:
                local_contributions = [{} for _ in range(num_threads)]
                
                # Distribute nodes across threads
                for i in p.range(n):
                    node = nodes[i]
                    thread_id = p.thread_num
                    
                    # Skip nodes with no outgoing edges (handled in normalization)
                    if out_degrees[node] > 0:
                        # Calculate contribution to each neighbor
                        contribution = pagerank[node] * damping / out_degrees[node]
                        
                        # Store contributions locally to avoid race conditions
                        for neighbor in adj_list[node]:
                            if neighbor not in local_contributions[thread_id]:
                                local_contributions[thread_id][neighbor] = 0.0
                            local_contributions[thread_id][neighbor] += contribution
            
            # Combine all contributions sequentially
            for thread_contrib in local_contributions:
                for node, contrib in thread_contrib.items():
                    new_pagerank[node] += contrib
                    
            # Handle dangling nodes (no outgoing edges)
            dangling_pr = sum(pagerank[node] for node in nodes if out_degrees[node] == 0)
            dangling_contrib = damping * dangling_pr / n
            for node in nodes:
                new_pagerank[node] += dangling_contrib
            
            # Check for convergence
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes)
            converged = diff < tol
            
            # Update PageRank scores for next iteration
            pagerank = new_pagerank.copy()
            iterations += 1
        
        # Normalize PageRank scores
        pagerank_sum = sum(pagerank.values())
        pagerank = {node: score / pagerank_sum for node, score in pagerank.items()}
        
        return pagerank, iterations

    @staticmethod
    def check_neighbors_for_mp(data):
        """
        Helper function for multiprocessing PageRank.
        Computes contribution of a set of nodes to their neighbors.
        
        Parameters:
        -----------
        data : tuple (nodes, pagerank, out_degrees, damping)
            nodes: List of nodes to process
            pagerank: Dictionary of current PageRank values
            out_degrees: Dictionary of outgoing edge counts
            damping: Damping factor
            
        Returns:
        --------
        dict
            Dictionary mapping nodes to their received contributions
        """
        nodes, pagerank, out_degrees, damping, adj_list = data
        contributions = {}
        
        for node in nodes:
            # Skip nodes with no outgoing edges
            if out_degrees[node] == 0:
                continue
            
            # Calculate contribution to each neighbor
            weight = pagerank[node] * damping / out_degrees[node]
            
            for neighbor in adj_list[node]:
                if neighbor not in contributions:
                    contributions[neighbor] = 0
                contributions[neighbor] += weight
        
        return contributions

    @staticmethod
    def traditional_pagerank_multiprocessing(adj_list, damping=0.85, max_iterations=100, tol=1e-6, num_processes=None):
        """
        Multiprocessing-accelerated PageRank implementation
        
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
        num_processes : int or None
            Number of processes to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        pagerank : dict
            Dictionary mapping node IDs to PageRank scores
        iterations : int
            Number of iterations performed
        """
        import multiprocessing as mp
        from functools import partial
        
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Get the list of nodes
        nodes = sorted(list(adj_list.keys()))
        n = len(nodes)
        
        # Calculate out-degrees
        out_degrees = {node: len(neighbors) for node, neighbors in adj_list.items()}
        
        # Initialize PageRank values
        pagerank = {node: 1.0 / n for node in nodes}
        
        # PageRank iteration
        iterations = 0
        converged = False
        
        while not converged and iterations < max_iterations:
            # Create a copy of current PageRank values
            old_pagerank = pagerank.copy()
            
            # Calculate dangling node contribution
            dangling_nodes = [node for node in nodes if out_degrees[node] == 0]
            dangling_sum = sum(old_pagerank[node] for node in dangling_nodes)
            dangling_contrib = damping * dangling_sum / n
            
            # Initialize new PageRank values with teleportation and dangling contributions
            pagerank = {node: (1.0 - damping) / n + dangling_contrib for node in nodes}
            
            # Divide nodes into chunks for parallel processing
            chunk_size = max(1, len(nodes) // num_processes)
            chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
            
            # Process node contributions in parallel
            with mp.Pool(processes=min(num_processes, len(chunks))) as pool:
                # Prepare data for each process
                data_chunks = [(chunk, old_pagerank, out_degrees, damping, adj_list) for chunk in chunks]
                
                # Map function across processes
                results = pool.map(PageRank.check_neighbors_for_mp, data_chunks)
                
                # Combine results from all processes
                for contrib_dict in results:
                    for node, contrib in contrib_dict.items():
                        pagerank[node] += contrib
            
            # Check for convergence
            diff = sum(abs(pagerank[node] - old_pagerank[node]) for node in nodes)
            converged = diff < tol
            
            iterations += 1
        
        # Normalize PageRank values to sum to 1
        pagerank_sum = sum(pagerank.values())
        pagerank = {node: rank / pagerank_sum for node, rank in pagerank.items()}
        
        return pagerank, iterations
        
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