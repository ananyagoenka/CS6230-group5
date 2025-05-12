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

# Import numba
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    print("WARNING: numba not found. Install with 'pip install numba' for Numba acceleration.")
    HAS_NUMBA = False

class ConnectedComponents:
    """
    Class containing both traditional and linear algebra implementations of Connected Components
    """
    
    @staticmethod
    def traditional_cc_cpu(adj_list):
        """
        Traditional Connected Components implementation using iterative DFS
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        visited = set()
        components = []
        component_map = {}
        component_id = 0
        
        # Iterative DFS
        for start_node in adj_list:
            if start_node in visited:
                continue
                
            # Start a new component
            component = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                
                if node in visited:
                    continue
                    
                # Mark as visited and add to current component
                visited.add(node)
                component.append(node)
                component_map[node] = component_id
                
                # Add unvisited neighbors to stack
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            # Add completed component to list
            components.append(component)
            component_id += 1
        
        return components, component_map
    
    @staticmethod
    def traditional_cc_openmp(adj_list, num_threads=None):
        """
        OpenMP-accelerated Connected Components implementation
        
        This algorithm uses a two-phase approach:
        1. First, sequentially identify connected components using iterative DFS
        2. Then, parallelize the processing within each component
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        num_threads : int or None
            Number of threads to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        if not HAS_PYMP:
            print("OpenMP not available, falling back to traditional CC")
            return ConnectedComponents.traditional_cc_cpu(adj_list)
            
        if num_threads is None:
            num_threads = mp.cpu_count()
        
        # Phase 1: Identify components sequentially
        # This uses the sequential algorithm to find connected components
        visited = set()
        components = []
        
        # Iterative DFS to identify components
        for start_node in adj_list:
            if start_node in visited:
                continue
                
            # Start a new component
            component = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                
                if node in visited:
                    continue
                    
                # Mark as visited and add to current component
                visited.add(node)
                component.append(node)
                
                # Add unvisited neighbors to stack
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            # Add completed component to list
            components.append(component)
        
        # Phase 2: Process each component in parallel
        component_map = {}  # Will be filled by parallel threads
        
        with pymp.Parallel(num_threads) as p:
            # Lock for updating the shared component_map
            component_map_lock = p.lock
            
            # Divide components among threads
            for comp_idx in p.range(len(components)):
                component = components[comp_idx]
                
                # Process each node in this component
                for node in component:
                    # Critical section: update component_map
                    with component_map_lock:
                        component_map[node] = comp_idx
                        
                    # Optionally do more processing here as needed
                    # This is where you would parallelize computation within the component
        
        return components, component_map

    @staticmethod
    def check_neighbors_for_mp(chunk, graph_data, results_data):
        """
        Helper function for multiprocessing CC.
        Process a chunk of nodes to update their component labels.
        
        Parameters:
        -----------
        chunk : list
            List of nodes to process
        graph_data : tuple
            Tuple containing (adj_list, labels)
        results_data : multiprocessing.Manager().dict
            Shared dictionary to store updated labels
            
        Returns:
        --------
        bool
            True if any labels were updated, False otherwise
        """
        adj_list, current_labels = graph_data
        updated = False
        
        for node in chunk:
            if node not in current_labels:
                continue
                
            # Get current label
            current_label = current_labels[node]
            
            # Find minimum label among neighbors
            neighbor_labels = [current_labels[neighbor] for neighbor in adj_list[node] 
                              if neighbor in current_labels]
            
            if neighbor_labels:
                min_label = min(neighbor_labels + [current_label])
                if min_label < current_label:
                    results_data[node] = min_label
                    updated = True
        
        return updated

    @staticmethod
    def traditional_cc_multiprocessing(adj_list, num_processes=None):
        """
        Multiprocessing-accelerated Connected Components implementation
        
        This algorithm uses label propagation:
        1. Initialize each node with a unique label
        2. Iteratively update labels until convergence
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
        num_processes : int or None
            Number of processes to use. If None, uses number of CPU cores.
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        from functools import partial
        import multiprocessing as mp
        
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        # Initialize each node with its own ID as label
        labels = {node: node for node in adj_list}
        
        # Create manager and shared dictionary
        manager = mp.Manager()
        shared_results = manager.dict()
        
        # Get the list of nodes
        nodes = list(adj_list.keys())
        
        # Split nodes into chunks
        chunk_size = max(1, len(nodes) // num_processes)
        chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]
        
        # Run label propagation until convergence
        converged = False
        while not converged:
            # Reset convergence flag and shared results
            converged = True
            shared_results.clear()
            
            # Process chunks in parallel
            with mp.Pool(processes=min(num_processes, len(chunks))) as pool:
                # Create partial function with fixed parameters
                process_func = partial(ConnectedComponents.check_neighbors_for_mp, 
                                      graph_data=(adj_list, labels),
                                      results_data=shared_results)
                
                # Map the function to chunks
                chunk_results = pool.map(process_func, chunks)
                
                # If any chunk returned True, we're not converged
                if any(chunk_results):
                    converged = False
            
            # Update labels with results from this iteration
            for node, new_label in shared_results.items():
                labels[node] = new_label
        
        # Extract components from final labels
        component_groups = defaultdict(list)
        for node, label in labels.items():
            component_groups[label].append(node)
        
        # Convert to the expected return format
        components = list(component_groups.values())
        component_map = labels.copy()
        
        return components, component_map

    @staticmethod
    def traditional_cc_numba(adj_list):
        """
        Numba-accelerated Connected Components implementation
        
        Parameters:
        -----------
        adj_list : dict
            Adjacency list representation of the graph
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        if not HAS_NUMBA:
            print("Numba not available, falling back to traditional CC")
            return ConnectedComponents.traditional_cc_cpu(adj_list)
        
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
        
        try:
            # Define Numba function for CC using label propagation
            @jit(nopython=True, parallel=True)
            def numba_cc(indptr, indices, n):
                # Initialize labels
                labels = np.arange(n, dtype=np.int64)
                
                # Run label propagation until convergence
                converged = False
                while not converged:
                    converged = True
                    
                    # For each node, propagate minimum label
                    for i in range(n):
                        min_label = labels[i]
                        
                        # Check neighbors
                        for j in range(indptr[i], indptr[i+1]):
                            neighbor = indices[j]
                            if labels[neighbor] < min_label:
                                min_label = labels[neighbor]
                        
                        # Update label if needed
                        if min_label < labels[i]:
                            labels[i] = min_label
                            converged = False
                
                # Second pass: compress paths
                # Ensure all nodes in a component have the same label
                changed = True
                while changed:
                    changed = False
                    for i in range(n):
                        for j in range(indptr[i], indptr[i+1]):
                            neighbor = indices[j]
                            if labels[i] > labels[neighbor]:
                                labels[i] = labels[neighbor]
                                changed = True
                
                return labels
            
            # Run Numba CC
            labels_array = numba_cc(indptr_array, indices_array, n)
        
        except Exception as e:
            print(f"Error in Numba CC: {e}")
            print("Falling back to traditional CC")
            return ConnectedComponents.traditional_cc_cpu(adj_list)
        
        # Convert back to original format
        component_map = {idx_to_node[i]: int(label) for i, label in enumerate(labels_array)}
        
        # Extract components from labels
        component_groups = defaultdict(list)
        for i, label in enumerate(labels_array):
            component_groups[int(label)].append(idx_to_node[i])
        
        components = list(component_groups.values())
        
        return components, component_map
        
    @staticmethod
    def la_cc_cpu(adj_matrix):
        """
        Linear algebra Connected Components implementation using adjacency matrix on CPU
        
        Uses matrix multiplication to find connected components.
        
        Parameters:
        -----------
        adj_matrix : numpy.ndarray
            Adjacency matrix representation of the graph
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        n = adj_matrix.shape[0]
        
        # Initialize reachability matrix with identity matrix
        # Add identity to handle self-loops properly
        reachability = adj_matrix + np.eye(n)
        
        # Compute transitive closure using matrix multiplication
        # R^(k+1) = R^k OR (R^k · R^k)
        # Continue until convergence
        converged = False
        while not converged:
            new_reachability = reachability.copy()
            
            # Matrix multiplication to find new paths
            new_reachability = np.matmul(reachability, reachability) > 0
            
            # Convert to boolean matrix
            new_reachability = new_reachability.astype(bool)
            
            # Check for convergence
            if np.array_equal(new_reachability, reachability):
                converged = True
            
            reachability = new_reachability
        
        # Extract connected components from the reachability matrix
        visited = np.zeros(n, dtype=bool)
        components = []
        component_map = {}
        
        for i in range(n):
            if not visited[i]:
                # Find all nodes reachable from i
                component = np.where(reachability[i])[0].tolist()
                components.append(component)
                
                # Update component map and mark nodes as visited
                for node in component:
                    component_map[node] = len(components) - 1
                    visited[node] = True
        
        return components, component_map
    
    @staticmethod
    def la_cc_gpu(adj_matrix):
        """
        Linear algebra Connected Components implementation using adjacency matrix on GPU
        
        Parameters:
        -----------
        adj_matrix : torch.Tensor
            Adjacency matrix representation of the graph on GPU
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        n = adj_matrix.shape[0]
        device = adj_matrix.device
        
        try:
            # Initialize reachability matrix with identity matrix
            # Add identity to handle self-loops properly
            identity = torch.eye(n, device=device)
            reachability = adj_matrix + identity
            
            # Convert to float tensor (for matrix multiplication)
            reachability = (reachability > 0).float()
            
            # Compute transitive closure using matrix multiplication
            # R^(k+1) = R^k OR (R^k · R^k)
            # Continue until convergence
            converged = False
            iterations = 0
            max_iterations = min(n, 100)  # Limit the number of iterations to prevent resource exhaustion
            
            while not converged and iterations < max_iterations:
                try:
                    # Matrix multiplication to find new paths (using float tensors)
                    new_reachability = torch.matmul(reachability, reachability)
                    
                    # Convert to binary values (0 or 1)
                    new_reachability = (new_reachability > 0).float()
                    
                    # Check for convergence
                    if torch.equal(new_reachability, reachability):
                        converged = True
                    
                    reachability = new_reachability
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "insufficient resources" in str(e):
                        print("Warning: GPU resources insufficient. Falling back to CPU...")
                        # Fall back to CPU implementation with the current state
                        reachability_cpu = reachability.cpu().numpy() > 0
                        break
                    else:
                        raise  # Re-raise if it's not a resource issue
                
                iterations += 1
            
            # If we exhausted iterations without converging, still proceed with what we have
            if iterations >= max_iterations and not converged:
                print("Warning: Maximum iterations reached before convergence. Results may be incomplete.")
            
            # If we fell back to CPU, compute the rest on CPU
            if 'reachability_cpu' in locals():
                # Complete the computation on CPU using numpy
                converged = False
                while not converged:
                    new_reachability_cpu = np.matmul(reachability_cpu, reachability_cpu) > 0
                    if np.array_equal(new_reachability_cpu, reachability_cpu):
                        converged = True
                    reachability_cpu = new_reachability_cpu
                
                # Extract components
                visited = np.zeros(n, dtype=bool)
                components = []
                component_map = {}
                
                for i in range(n):
                    if not visited[i]:
                        component = np.where(reachability_cpu[i])[0].tolist()
                        components.append(component)
                        for node in component:
                            component_map[node] = len(components) - 1
                            visited[node] = True
                
                return components, component_map
                
            # Continue with GPU computation if we didn't fall back
            # Convert final result to boolean for component extraction
            reachability_bool = reachability.bool()
            
            # Extract connected components from the reachability matrix
            visited = torch.zeros(n, dtype=torch.bool, device=device)
            components = []
            component_map = {}
            
            for i in range(n):
                if not visited[i]:
                    # Find all nodes reachable from i
                    component = torch.where(reachability_bool[i])[0].cpu().numpy().tolist()
                    components.append(component)
                    
                    # Update component map and mark nodes as visited
                    for node in component:
                        component_map[node] = len(components) - 1
                        visited[node] = True
            
            return components, component_map
            
        except RuntimeError as e:
            print(f"GPU error: {e}")
            print("Falling back to CPU implementation...")
            
            # Convert the adjacency matrix to CPU and use the CPU implementation
            adj_matrix_np = adj_matrix.cpu().numpy()
            return ConnectedComponents.la_cc_cpu(adj_matrix_np)
    
    @staticmethod
    def la_cc_sparse_gpu(adj_matrix):
        """
        Linear algebra Connected Components implementation using sparse adjacency matrix on GPU
        This is optimized for large sparse graphs
        
        Parameters:
        -----------
        adj_matrix : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph on GPU
            
        Returns:
        --------
        components : list
            List of components, each represented as a list of nodes
        component_map : dict
            Dictionary mapping each node to its component ID
        """
        n = adj_matrix.shape[0]
        device = adj_matrix.device
        
        try:
            # Initialize reachability matrix with identity matrix
            # Add identity to handle self-loops properly
            identity = torch.eye(n, device=device).to_sparse()
            reachability = adj_matrix + identity
            
            # Convert to float sparse tensor (for sparse matrix multiplication)
            reachability = reachability.coalesce()
            float_values = torch.ones_like(reachability.values(), dtype=torch.float32)
            reachability = torch.sparse.FloatTensor(
                reachability.indices(), float_values, reachability.size()
            ).coalesce()
            
            # Compute transitive closure using matrix multiplication
            # R^(k+1) = R^k OR (R^k · R^k)
            # Continue until convergence
            converged = False
            iterations = 0
            max_iterations = min(n, 100)  # Limit the number of iterations to prevent resource exhaustion
            
            while not converged and iterations < max_iterations:
                try:
                    # Sparse matrix multiplication
                    new_reachability = torch.sparse.mm(reachability, reachability)
                    
                    # Convert to binary values (0 or 1) while keeping sparse format
                    new_reachability = new_reachability.coalesce()
                    float_values = torch.ones_like(new_reachability.values(), dtype=torch.float32)
                    new_reachability = torch.sparse.FloatTensor(
                        new_reachability.indices(), float_values, new_reachability.size()
                    ).coalesce()
                    
                    # Check for convergence - compare with current reachability
                    # We'll use the indices pattern to check for identical sparsity patterns
                    current_indices = set(tuple(idx) for idx in reachability.indices().t().cpu().numpy())
                    new_indices = set(tuple(idx) for idx in new_reachability.indices().t().cpu().numpy())
                    
                    if current_indices == new_indices:
                        converged = True
                    
                    reachability = new_reachability
                    
                except RuntimeError as e:
                    if "insufficient resources" in str(e) or "out of memory" in str(e):
                        print("Warning: GPU resources insufficient for sparse operation. Falling back to CPU...")
                        # Fall back to CPU implementation with the current state
                        reachability_cpu = reachability.cpu().to_dense().numpy() > 0
                        break
                    else:
                        raise  # Re-raise if it's not a resource issue
                
                iterations += 1
            
            # If we exhausted iterations without converging, still proceed with what we have
            if iterations >= max_iterations and not converged:
                print("Warning: Maximum iterations reached before convergence. Results may be incomplete.")
            
            # If we fell back to CPU, compute the rest on CPU
            if 'reachability_cpu' in locals():
                # Complete the computation on CPU using numpy
                converged = False
                while not converged:
                    new_reachability_cpu = np.matmul(reachability_cpu, reachability_cpu) > 0
                    if np.array_equal(new_reachability_cpu, reachability_cpu):
                        converged = True
                    reachability_cpu = new_reachability_cpu
                
                # Extract components
                visited = np.zeros(n, dtype=bool)
                components = []
                component_map = {}
                
                for i in range(n):
                    if not visited[i]:
                        component = np.where(reachability_cpu[i])[0].tolist()
                        components.append(component)
                        for node in component:
                            component_map[node] = len(components) - 1
                            visited[node] = True
                
                return components, component_map
            
            # Continue with GPU computation if we didn't fall back
            # Convert to dense format for component extraction
            reachability_dense = reachability.to_dense().bool()
            
            # Extract connected components
            visited = torch.zeros(n, dtype=torch.bool, device=device)
            components = []
            component_map = {}
            
            for i in range(n):
                if not visited[i]:
                    # Find all nodes reachable from i
                    component = torch.where(reachability_dense[i])[0].cpu().numpy().tolist()
                    components.append(component)
                    
                    # Update component map and mark nodes as visited
                    for node in component:
                        component_map[node] = len(components) - 1
                        visited[node] = True
            
            return components, component_map
            
        except RuntimeError as e:
            print(f"GPU error: {e}")
            print("Falling back to CPU implementation...")
            
            # Convert the adjacency matrix to CPU and use the CPU implementation
            adj_matrix_np = adj_matrix.cpu().to_dense().numpy()
            return ConnectedComponents.la_cc_cpu(adj_matrix_np)