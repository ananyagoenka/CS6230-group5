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
        
        This implementation focuses on parallelizing specific parts of the BFS
        algorithm while maintaining the exact same output as the sequential version.
        
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
        
        # Initialize distances with infinity
        distances = {node: float('infinity') for node in adj_list}
        distances[start_node] = 0
        
        # Initialize visited list and set for quick lookup
        visited = [start_node]
        visited_set = set([start_node])
        
        # Initialize queue for BFS
        queue = deque([start_node])
        
        # Process the BFS queue
        while queue:
            # Get the next node to process
            node = queue.popleft()
            
            # Get neighbors
            neighbors = list(adj_list[node])
            
            # Use OpenMP to check neighbors in parallel
            # Create a filtered list of unvisited neighbors
            unvisited_neighbors = []
            
            # Only parallelize if we have enough neighbors to make it worthwhile
            if len(neighbors) >= 4:  # Arbitrary threshold for parallelization
                with pymp.Parallel(min(num_threads, len(neighbors))) as p:
                    # Create thread-local lists for collecting unvisited neighbors
                    local_unvisited = [[] for _ in range(p.num_threads)]
                    
                    # Split neighbors among threads
                    for i in p.range(len(neighbors)):
                        neighbor = neighbors[i]
                        thread_id = p.thread_num
                        
                        # This is a read-only operation on the visited_set,
                        # so it's safe to do in parallel without locks
                        if neighbor not in visited_set:
                            local_unvisited[thread_id].append(neighbor)
                    
                    # Combine results from all threads
                    for local_list in local_unvisited:
                        unvisited_neighbors.extend(local_list)
            else:
                # For small neighbor lists, just process sequentially
                unvisited_neighbors = [n for n in neighbors if n not in visited_set]
            
            # Process the unvisited neighbors sequentially to maintain BFS order
            current_distance = distances[node]
            for neighbor in unvisited_neighbors:
                # Double-check that it's not already visited (may have changed)
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    visited.append(neighbor)
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        return visited, distances

    @staticmethod
    def check_neighbors_for_mp(chunk, visited_set):
        """
        Helper function for multiprocessing BFS.
        Checks which neighbors in a chunk are not in the visited set.
        
        Parameters:
        -----------
        chunk : list
            List of nodes to check
        visited_set : set
            Set of already visited nodes
            
        Returns:
        --------
        list
            List of unvisited nodes from the chunk
        """
        return [n for n in chunk if n not in visited_set]

    @staticmethod
    def traditional_bfs_multiprocessing(adj_list, start_node, num_processes=None):
        """
        Multiprocessing-accelerated BFS implementation
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        start_node : int
            Starting node for BFS
        num_processes : int or None
            Number of processes to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        import multiprocessing as mp
        from collections import deque
        from functools import partial
        
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Initialize distances with infinity
        distances = {node: float('infinity') for node in adj_list}
        distances[start_node] = 0
        
        # Initialize visited list and set for quick lookup
        visited = [start_node]
        visited_set = set([start_node])
        
        # Initialize queue for BFS
        queue = deque([start_node])
        
        # Process the BFS queue
        while queue:
            # Get the next node to process
            node = queue.popleft()
            
            # Get neighbors
            neighbors = list(adj_list[node])
            
            # For small sets, just do it sequentially
            if len(neighbors) < 100:  # Arbitrary threshold - multiprocessing has higher overhead
                unvisited_neighbors = [n for n in neighbors if n not in visited_set]
            else:
                # Split neighbors into chunks
                chunk_size = max(1, len(neighbors) // num_processes)
                chunks = [neighbors[i:i+chunk_size] for i in range(0, len(neighbors), chunk_size)]
                
                # Process chunks in parallel
                with mp.Pool(processes=min(num_processes, len(chunks))) as pool:
                    # Use a partial function to pass the visited set
                    check_func = partial(BFS.check_neighbors_for_mp, visited_set=visited_set)
                    
                    # Map the function to all chunks
                    results = pool.map(check_func, chunks)
                    
                    # Combine results
                    unvisited_neighbors = []
                    for result in results:
                        unvisited_neighbors.extend(result)
            
            # Process the unvisited neighbors sequentially
            current_distance = distances[node]
            for neighbor in unvisited_neighbors:
                if neighbor not in visited_set:  # Double-check
                    visited_set.add(neighbor)
                    visited.append(neighbor)
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        return visited, distances

    @staticmethod
    def traditional_bfs_numba(adj_list, start_node):
        """
        Numba-accelerated BFS implementation
        
        Note: This requires converting the graph to a numerical representation
        
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
        # Try to import numba
        try:
            import numba
            from numba import jit, prange
            import numpy as np
            has_numba = True
        except ImportError:
            print("Numba not available, falling back to traditional BFS")
            has_numba = False
        
        if not has_numba:
            return BFS.traditional_bfs_cpu(adj_list, start_node)
        
        # Convert graph to CSR format for Numba
        nodes = sorted(adj_list.keys())
        n = len(nodes)
        
        # Create mapping between original node IDs and indices
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        # Create CSR representation
        indptr = [0]
        indices = []
        for node in nodes:
            indices.extend([node_to_idx[neighbor] for neighbor in adj_list[node]])
            indptr.append(len(indices))
        
        # Convert to numpy arrays
        indptr_array = np.array(indptr, dtype=np.int64)
        indices_array = np.array(indices, dtype=np.int64)
        start_idx = node_to_idx[start_node]
        
        try:
            # Define Numba function for BFS
            @jit(nopython=True, parallel=True)
            def numba_bfs(indptr, indices, start_idx, n):
                # Initialize arrays
                distances = np.full(n, np.inf)
                distances[start_idx] = 0
                visited = np.zeros(n, dtype=np.bool_)
                visited[start_idx] = True
                
                # Initialize frontier
                frontier = np.zeros(n, dtype=np.bool_)
                frontier[start_idx] = True
                
                level = 0
                while np.any(frontier):
                    level += 1
                    new_frontier = np.zeros(n, dtype=np.bool_)
                    
                    # Process frontier in parallel
                    for i in prange(n):
                        if frontier[i]:
                            # Get neighbors
                            for j in range(indptr[i], indptr[i+1]):
                                neighbor = indices[j]
                                # Check and set atomically
                                if not visited[neighbor]:
                                    visited[neighbor] = True
                                    distances[neighbor] = level
                                    new_frontier[neighbor] = True
                    
                    frontier = new_frontier
                
                return distances, visited
            
            # Run Numba BFS
            distances_arr, visited_arr = numba_bfs(indptr_array, indices_array, start_idx, n)
        
        except Exception as e:
            print(f"Error in Numba BFS: {e}")
            print("Falling back to traditional BFS")
            return BFS.traditional_bfs_cpu(adj_list, start_node)
        
        # Convert back to original format
        distances = {idx_to_node[i]: float(dist) if dist != np.inf else float('infinity') 
                    for i, dist in enumerate(distances_arr)}
        
        # Create visited list in BFS order
        visited_indices = np.where(visited_arr)[0]
        visited_by_distance = sorted([(i, distances_arr[i]) for i in visited_indices 
                                    if distances_arr[i] != np.inf], 
                                    key=lambda x: x[1])
        visited = [idx_to_node[i] for i, _ in visited_by_distance]
        
        # Ensure start node is the first in the visited list
        if visited and visited[0] != start_node:
            visited.remove(start_node)
            visited.insert(0, start_node)
        
        return visited, distances
        
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