import numpy as np
import torch
import time
from collections import deque, defaultdict
import multiprocessing as mp

# Import OpenMP Python bindings
try:
    import pymp
    HAS_PYMP = True
except ImportError:
    print("WARNING: pymp-pypi not found. Install with 'pip install pymp-pypi' for OpenMP support.")
    HAS_PYMP = False

class BFS:
    """
    Class containing both traditional and linear algebra implementations of BFS
    """
    
    @staticmethod
    def traditional_bfs_cpu(adj_list, start_node):
        """
        Traditional BFS implementation using adjacency list
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        start_node : int
            Starting node for BFS
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        visited = []
        distances = {node: float('infinity') for node in adj_list}
        distances[start_node] = 0
        
        q = deque([start_node])
        
        while q:
            node = q.popleft()
            visited.append(node)
            
            for neighbor in adj_list[node]:
                if distances[neighbor] == float('infinity'):
                    distances[neighbor] = distances[node] + 1
                    q.append(neighbor)
        
        return visited, distances
    
    @staticmethod
    def traditional_bfs_openmp(adj_list, start_node, num_threads=None):
        """
        OpenMP-accelerated BFS implementation using adjacency list
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        start_node : int
            Starting node for BFS
        num_threads : int or None
            Number of threads to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        if not HAS_PYMP:
            print("OpenMP not available, falling back to traditional BFS")
            return BFS.traditional_bfs_cpu(adj_list, start_node)
            
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        # Create a list representation of the graph for more efficient sharing
        nodes = list(adj_list.keys())
        n = len(nodes)
        
        # Initialize distances with infinity
        distances = {}
        for node in nodes:
            distances[node] = float('infinity')
        distances[start_node] = 0
        
        # Initialize visited and frontier structures
        visited_arr = np.zeros(n, dtype=bool)  # Boolean array for efficient checks
        node_to_idx = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices
        idx_to_node = {i: node for i, node in enumerate(nodes)}  # Map indices to nodes
        
        visited_idx = node_to_idx[start_node]
        visited_arr[visited_idx] = True
        
        current_frontier = [start_node]
        visited_list = [start_node]
        
        level = 0
        
        # BFS traversal
        while current_frontier:
            next_frontier = []
            level += 1
            
            # Process current frontier in parallel with OpenMP
            with pymp.Parallel(num_threads) as p:
                # Create thread-local next frontier lists
                local_next_frontiers = [[] for _ in range(num_threads)]
                
                # Distribute nodes across threads
                for i in p.range(len(current_frontier)):
                    node = current_frontier[i]
                    thread_id = p.thread_num
                    
                    for neighbor in adj_list[node]:
                        neighbor_idx = node_to_idx[neighbor]
                        
                        # First check without lock (optimistic)
                        if not visited_arr[neighbor_idx]:
                            # Use critical section for updating shared state
                            with p.lock:
                                if not visited_arr[neighbor_idx]:  # Double-check
                                    visited_arr[neighbor_idx] = True
                                    distances[neighbor] = level
                                    local_next_frontiers[thread_id].append(neighbor)
                
                # Merge local frontiers into global next frontier
                with p.lock:
                    for local_frontier in local_next_frontiers:
                        next_frontier.extend(local_frontier)
                        visited_list.extend(local_frontier)
            
            # Update frontier for next iteration
            current_frontier = next_frontier
        
        return visited_list, distances
    
    @staticmethod
    def la_bfs_cpu(adj_matrix, start_node):
        """
        Linear algebra BFS implementation using adjacency matrix on CPU
        
        Parameters:
        -----------
        adj_matrix : numpy.ndarray
            Adjacency matrix representation of the graph
        start_node : int
            Starting node for BFS
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        n = adj_matrix.shape[0]
        visited = []
        distances = {node: float('infinity') for node in range(n)}
        distances[start_node] = 0
        
        # Initialize frontier vector (one-hot encoding of start node)
        frontier = np.zeros(n)
        frontier[start_node] = 1
        
        # Initialize visited vector
        visited_vec = np.zeros(n)
        visited_vec[start_node] = 1
        
        level = 0
        while np.any(frontier):
            # Add current frontier to visited list
            current_nodes = np.where(frontier)[0]
            visited.extend(current_nodes)
            
            # Update distances for current frontier
            for node in current_nodes:
                distances[node] = level
            
            # Update frontier: f_{i+1} = (A · f_i) ∧ ¬v_i
            # Matrix multiplication to find neighbors
            frontier = np.matmul(adj_matrix, frontier)
            
            # Only keep unvisited nodes
            frontier = np.logical_and(frontier > 0, np.logical_not(visited_vec))
            
            # Update visited vector
            visited_vec = np.logical_or(visited_vec, frontier)
            
            # Convert to proper numeric type
            frontier = frontier.astype(np.float64)
            
            level += 1
        
        return visited, distances
    
    @staticmethod
    def la_bfs_gpu(adj_matrix, start_node):
        """
        Linear algebra BFS implementation using adjacency matrix on GPU
        
        Parameters:
        -----------
        adj_matrix : torch.Tensor
            Adjacency matrix representation of the graph on GPU
        start_node : int
            Starting node for BFS
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        n = adj_matrix.shape[0]
        visited = []
        distances = {node: float('infinity') for node in range(n)}
        distances[start_node] = 0
        
        # Initialize frontier vector (one-hot encoding of start node)
        frontier = torch.zeros(n, device=adj_matrix.device)
        frontier[start_node] = 1
        
        # Initialize visited vector
        visited_vec = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
        visited_vec[start_node] = True
        
        level = 0
        while torch.any(frontier > 0):
            # Add current frontier to visited list
            current_nodes = torch.where(frontier > 0)[0].cpu().numpy()
            visited.extend(current_nodes)
            
            # Update distances for current frontier
            for node in current_nodes:
                distances[node] = level
            
            # Update frontier: f_{i+1} = (A · f_i) ∧ ¬v_i
            # Matrix multiplication to find neighbors
            frontier = torch.matmul(adj_matrix, frontier)
            
            # Only keep unvisited nodes
            frontier = (frontier > 0) & (~visited_vec)
            
            # Update visited vector
            visited_vec = visited_vec | frontier
            
            # Convert to proper numeric type
            frontier = frontier.float()
            
            level += 1
        
        return visited, distances
    
    @staticmethod
    def la_bfs_sparse_gpu(adj_matrix, start_node):
        """
        Linear algebra BFS implementation using sparse adjacency matrix on GPU
        This is optimized for large sparse graphs
        
        Parameters:
        -----------
        adj_matrix : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph on GPU
        start_node : int
            Starting node for BFS
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        n = adj_matrix.shape[0]
        visited = []
        distances = {node: float('infinity') for node in range(n)}
        distances[start_node] = 0
        
        # Initialize frontier vector (one-hot encoding of start node)
        frontier = torch.zeros(n, device=adj_matrix.device)
        frontier[start_node] = 1
        
        # Initialize visited vector
        visited_vec = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
        visited_vec[start_node] = True
        
        level = 0
        while torch.any(frontier > 0):
            # Add current frontier to visited list
            current_nodes = torch.where(frontier > 0)[0].cpu().numpy()
            visited.extend(current_nodes)
            
            # Update distances for current frontier
            for node in current_nodes:
                distances[node] = level
            
            # Update frontier: f_{i+1} = (A · f_i) ∧ ¬v_i
            # Use sparse matrix multiplication for efficiency
            frontier = torch.sparse.mm(adj_matrix, frontier.unsqueeze(1)).squeeze(1)
            
            # Only keep unvisited nodes
            frontier = (frontier > 0) & (~visited_vec)
            
            # Update visited vector
            visited_vec = visited_vec | frontier
            
            # Convert to proper numeric type
            frontier = frontier.float()
            
            level += 1
        
        return visited, distances