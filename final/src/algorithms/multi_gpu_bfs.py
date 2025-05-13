import numpy as np
import torch
import time
from collections import deque, defaultdict
import torch.distributed as dist
import os
import gc

class MultiGPUBFS:
    """
    Class containing multi-GPU implementation of BFS using graph partitioning
    """
    
    @staticmethod
    def initialize_distributed(backend='nccl'):
        """
        Initialize PyTorch distributed environment
        
        Parameters:
        -----------
        backend : str
            Backend to use for distributed operations (default: 'nccl')
        """
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend=backend)
                print(f"Initialized distributed environment with {dist.get_world_size()} processes")
                print(f"Current rank: {dist.get_rank()}")
            except Exception as e:
                print(f"Error initializing distributed environment: {e}")
                raise

    @staticmethod
    def finalize_distributed():
        """
        Clean up distributed environment
        """
        if dist.is_initialized():
            dist.destroy_process_group()
    
    @staticmethod
    def partition_graph(adj_matrix_sparse, num_partitions):
        """
        Partition graph by rows for multi-GPU processing
        Uses a simple row-based partitioning strategy
        
        Parameters:
        -----------
        adj_matrix_sparse : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph
        num_partitions : int
            Number of partitions to create
            
        Returns:
        --------
        list of torch.sparse.Tensor
            List of partitioned adjacency matrices
        """
        n = adj_matrix_sparse.shape[0]
        
        # Calculate partition sizes
        partition_size = n // num_partitions
        remainder = n % num_partitions
        
        partitions = []
        start_idx = 0
        
        for i in range(num_partitions):
            # Add extra row to earlier partitions if needed
            current_size = partition_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            
            # Extract rows for current partition
            indices = adj_matrix_sparse._indices()
            values = adj_matrix_sparse._values()
            
            # Filter indices for current partition
            mask = (indices[0] >= start_idx) & (indices[0] < end_idx)
            part_indices = indices[:, mask]
            part_values = values[mask]
            
            # Adjust row indices to be relative to partition
            part_indices[0] -= start_idx
            
            # Create sparse tensor for this partition
            part_matrix = torch.sparse.FloatTensor(
                part_indices, 
                part_values, 
                size=(current_size, n)
            )
            
            partitions.append(part_matrix)
            start_idx = end_idx
        
        return partitions

    @staticmethod
    def multi_gpu_bfs_distributed(adj_matrix_sparse, start_node):
        """
        Distributed BFS implementation using multiple GPUs
        
        Parameters:
        -----------
        adj_matrix_sparse : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph
        start_node : int
            Starting node for BFS
                
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        # Ensure distributed environment is initialized
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized. Call initialize_distributed() first.")
        
        # Get rank and world size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Master process (rank 0) coordinates
        if rank == 0:
            # Original matrix dimensions
            n = adj_matrix_sparse.shape[0]
            
            # Partition graph
            partitions = MultiGPUBFS.partition_graph(adj_matrix_sparse, world_size)
            
            # Send partitions to worker processes
            for i in range(1, world_size):
                # Serialize partition
                part_indices = partitions[i]._indices()
                part_values = partitions[i]._values()
                part_size = torch.tensor(partitions[i].size())
                
                # Send partition info
                dist.send(part_size, dst=i)
                dist.send(part_indices, dst=i)
                dist.send(part_values, dst=i)
            
            # Keep master's partition
            local_adj_matrix = partitions[0]
            local_device = local_adj_matrix.device
            local_size = local_adj_matrix.size()[0]
            
        else:  # Worker processes
            # Receive partition from master
            size_tensor = torch.tensor([0, 0], dtype=torch.long)
            dist.recv(size_tensor, src=0)
            
            # Extract partition dimensions
            local_size = size_tensor[0].item()
            n = size_tensor[1].item()
            
            # Allocate tensors for indices and values
            max_nnz = n * local_size  # Conservative upper bound
            indices = torch.zeros(2, max_nnz, dtype=torch.long)
            
            # Receive actual nnz count
            nnz_tensor = torch.tensor([0], dtype=torch.long)
            dist.recv(nnz_tensor, src=0)
            nnz = nnz_tensor[0].item()
            
            # Resize tensors to actual nnz
            indices = indices[:, :nnz]
            values = torch.zeros(nnz, dtype=torch.float)
            
            # Receive partition data
            dist.recv(indices, src=0)
            dist.recv(values, src=0)
            
            # Create sparse tensor
            local_device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
            local_adj_matrix = torch.sparse.FloatTensor(
                indices, values, size=(local_size, n)
            ).to(local_device)
        
        # Initialize data structures for BFS
        # Each GPU handles its local partition of visited/frontier
        visited_global = torch.zeros(n, dtype=torch.bool, device='cuda:0')
        distances_global = torch.full((n,), float('inf'), dtype=torch.float32, device='cuda:0')
        
        if rank == 0:
            # Set start node as visited and distance 0
            visited_global[start_node] = True
            distances_global[start_node] = 0
            
            # Initial frontier contains only start node
            frontier_global = torch.zeros(n, dtype=torch.bool, device='cuda:0')
            frontier_global[start_node] = True
        else:
            frontier_global = torch.zeros(n, dtype=torch.bool, device='cuda:0')
        
        # Share initial frontier and visited status
        dist.broadcast(visited_global, src=0)
        dist.broadcast(distances_global, src=0)
        dist.broadcast(frontier_global, src=0)
        
        # Copy to local device
        frontier_local = frontier_global.to(local_device)
        visited_local = visited_global.to(local_device)
        
        # Track nodes in BFS order
        visited_list = [start_node] if start_node < n else []
        
        # Current BFS level
        level = 0
        
        # Main BFS loop
        while torch.any(frontier_global):
            # Each GPU computes its part of the next frontier
            new_frontier_local = torch.zeros(n, dtype=torch.bool, device=local_device)
            
            # Neighbors of current frontier within local partition
            # Use sparse matrix multiplication
            neighbors = torch.sparse.mm(local_adj_matrix, frontier_local.float().view(-1, 1)).view(-1)
            
            # Find new unvisited nodes
            new_nodes_mask = (neighbors > 0) & (~visited_local)
            new_frontier_local[new_nodes_mask] = True
            
            # Gather new frontiers from all processes
            new_frontier_global = torch.zeros(n, dtype=torch.bool, device='cuda:0')
            dist.all_reduce(new_frontier_local.to('cuda:0'), new_frontier_global, op=dist.ReduceOp.MAX)
            
            # Increment level
            level += 1
            
            # Update visited and distances for new nodes
            if torch.any(new_frontier_global):
                new_nodes_indices = torch.nonzero(new_frontier_global).flatten()
                distances_global[new_nodes_indices] = level
                visited_global[new_nodes_indices] = True
                
                # Update visited list if on rank 0
                if rank == 0:
                    visited_list.extend(new_nodes_indices.cpu().numpy())
            
            # Update frontier for next iteration
            frontier_global = new_frontier_global
            frontier_local = frontier_global.to(local_device)
            visited_local = visited_global.to(local_device)
            
            # Break if no new nodes
            if not torch.any(new_frontier_global):
                break
            
            # Synchronize all processes
            dist.barrier()
        
        # Convert distance tensor to dictionary format
        if rank == 0:
            distances = {int(i): float(dist) if dist != float('inf') else float('infinity') 
                        for i, dist in enumerate(distances_global.cpu().numpy())}
            return visited_list, distances
        else:
            return None, None
    
    @staticmethod
    def la_bfs_multi_gpu_v2(adj_matrix_sparse, start_node, num_gpus=None):
        """
        Multi-GPU BFS implementation for single-node multi-GPU setup (simpler version)
        
        Parameters:
        -----------
        adj_matrix_sparse : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph
        start_node : int
            Starting node for BFS
        num_gpus : int or None
            Number of GPUs to use. If None, uses all available GPUs.
                
        Returns:
        --------
        visited : list
            List of nodes in BFS order
        distances : dict
            Dictionary of distances from start_node to each node
        """
        # Determine number of GPUs to use
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = min(num_gpus, torch.cuda.device_count())
        
        if num_gpus <= 0:
            raise RuntimeError("No GPUs available for multi-GPU BFS")
        
        if num_gpus == 1:
            # Fall back to single GPU version if only one GPU
            print("Using optimized single-GPU implementation for num_gpus=1")
            from src.algorithms.bfs import BFS
            return BFS.la_bfs_sparse_gpu_optimized_v2_turbo(adj_matrix_sparse, start_node)
        
        # Original matrix dimensions
        n = adj_matrix_sparse.shape[0]
        
        # Get graph size
        graph_size = n
        
        # For small graphs, fall back to single GPU version for better performance
        if graph_size <= 50000:  # Threshold based on testing
            print(f"Graph size {graph_size} <= 50000 nodes, using optimized single-GPU implementation")
            from src.algorithms.bfs import BFS
            return BFS.la_bfs_sparse_gpu_optimized_v2_turbo(adj_matrix_sparse, start_node)
        
        # Start with a full copy of the matrix for better performance in strong scaling
        # This approach is more efficient for small matrices and has better load balance
        # All GPUs get a full copy of the matrix
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        adj_matrices = []
        
        for i, device in enumerate(devices):
            # Move a copy of the adjacency matrix to each GPU
            adj_matrices.append(adj_matrix_sparse.to(device))
        
        # Initialize data structures for BFS
        distances = {node: float('infinity') for node in range(n)}
        distances[start_node] = 0
        
        visited = [start_node]
        
        # Initialize visited mask
        visited_mask = torch.zeros(n, dtype=torch.bool)
        visited_mask[start_node] = True
        
        # Initialize frontier (nodes to expand in the current iteration)
        frontier = torch.zeros(n, dtype=torch.bool)
        frontier[start_node] = True
        
        # Copy to main GPU (cuda:0)
        frontier_main = frontier.to('cuda:0')
        visited_mask_main = visited_mask.to('cuda:0')
        
        # Current BFS level
        level = 0
        
        # For load balancing, assign chunks of the frontier to different GPUs
        def get_node_chunks(total_nodes, num_chunks):
            """Split nodes into chunks for processing across GPUs"""
            chunk_size = total_nodes // num_chunks
            remainder = total_nodes % num_chunks
            
            chunks = []
            start = 0
            for i in range(num_chunks):
                size = chunk_size + (1 if i < remainder else 0)
                end = start + size
                chunks.append((start, end))
                start = end
                
            return chunks
        
        # Main BFS loop
        while torch.any(frontier_main):
            # Increment level
            level += 1
            
            # Get chunks to distribute work across GPUs
            chunks = get_node_chunks(n, num_gpus)
            
            # Process chunks on their respective GPUs
            new_frontiers = []
            for i, (start_idx, end_idx) in enumerate(chunks):
                # Create a mask for this chunk
                chunk_mask = torch.zeros(n, dtype=torch.bool, device=devices[i])
                chunk_mask[start_idx:end_idx] = True
                
                # Get frontier and visited mask for this GPU
                frontier_gpu = frontier_main.to(devices[i])
                visited_gpu = visited_mask_main.to(devices[i])
                
                # We only process the assigned chunk of nodes
                frontier_chunk = frontier_gpu & chunk_mask
                
                # If no frontier nodes in this chunk, skip processing
                if not torch.any(frontier_chunk):
                    new_frontiers.append(torch.zeros(n, dtype=torch.bool, device=devices[i]))
                    continue
                
                # Identify neighbors of frontier nodes
                neighbors = torch.sparse.mm(adj_matrices[i], frontier_chunk.float().view(-1, 1)).view(-1)
                
                # Find new unvisited nodes
                new_nodes = (neighbors > 0) & (~visited_gpu)
                
                # Add to new frontiers
                new_frontiers.append(new_nodes)
            
            # Combine results from all GPUs
            new_frontier = torch.zeros(n, dtype=torch.bool, device='cuda:0')
            for i, nf in enumerate(new_frontiers):
                new_frontier |= nf.to('cuda:0')
            
            # If no new nodes, break
            if not torch.any(new_frontier):
                break
            
            # Update visited mask and distances
            new_nodes_indices = torch.nonzero(new_frontier).flatten().cpu().numpy()
            
            # Update visited list
            visited.extend(new_nodes_indices)
            
            # Update distances for new nodes
            for node_idx in new_nodes_indices:
                distances[node_idx] = level
            
            # Update visited mask and frontier
            visited_mask_main[new_frontier] = True
            frontier_main = new_frontier
        
        return visited, distances