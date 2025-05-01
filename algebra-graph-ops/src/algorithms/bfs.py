import numpy as np
import torch
import time
import queue
from collections import deque
from scipy.sparse import csr_matrix

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