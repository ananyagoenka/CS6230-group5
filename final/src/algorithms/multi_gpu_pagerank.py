import numpy as np
import torch
import time
import torch.distributed as dist
import os
import gc

class MultiGPUPageRank:
    """
    Class containing multi-GPU implementation of PageRank using graph partitioning
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
    def multi_gpu_pagerank_distributed(adj_matrix_sparse, damping=0.85, max_iterations=100, tol=1e-6):
        """
        Distributed PageRank implementation using multiple GPUs
        
        Parameters:
        -----------
        adj_matrix_sparse : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph
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
            partitions = MultiGPUPageRank.partition_graph(adj_matrix_sparse, world_size)
            
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
        
        # Calculate out-degree sums
        local_out_degrees = torch.zeros(n, device=local_device)
        global_out_degrees = torch.zeros(n, device='cuda:0')
        
        # Sum over columns in local partition
        for i in range(local_size):
            local_start_idx = 0 if rank == 0 else sum(p.size(0) for p in partitions[:rank])
            global_idx = local_start_idx + i
            row_sum = torch.sparse.sum(local_adj_matrix[i]).item()
            local_out_degrees[global_idx] = row_sum
        
        # Collect out-degrees from all processes
        dist.all_reduce(local_out_degrees.to('cuda:0'), global_out_degrees, op=dist.ReduceOp.SUM)
        
        # Calculate inverse out-degrees
        inv_out_degrees = torch.zeros_like(global_out_degrees)
        mask = global_out_degrees > 0
        inv_out_degrees[mask] = 1.0 / global_out_degrees[mask]
        
        # Find dangling nodes (no outgoing edges)
        dangling_mask = ~mask
        
        # Initialize rank vector (uniform distribution)
        ranks_global = torch.full((n,), 1.0 / n, device='cuda:0')
        
        # Transpose adjacency matrix for more efficient operations
        local_adj_matrix_t = local_adj_matrix.transpose(0, 1).coalesce()
        
        # Prepare for iteration
        iterations = 0
        converged = False
        
        # Main iteration loop
        while not converged and iterations < max_iterations:
            # Keep old ranks for convergence check
            old_ranks = ranks_global.clone()
            
            # Calculate dangling weight contribution
            dangling_sum = torch.sum(ranks_global[dangling_mask])
            dangling_contribution = damping * dangling_sum / n
            
            # Copy current ranks to all devices
            ranks_local = ranks_global.to(local_device)
            
            # Apply inverse out-degrees
            scaled_ranks = ranks_local * inv_out_degrees.to(local_device)
            
            # Each GPU handles its portion of the graph
            local_new_ranks = torch.zeros(local_size, device=local_device)
            
            # Sparse matrix-vector multiplication for local partition
            local_new_ranks = torch.sparse.mm(local_adj_matrix_t, scaled_ranks.unsqueeze(1)).squeeze(1)
            
            # Scale by damping factor
            local_new_ranks *= damping
            
            # Gather results from all GPUs
            gathered_ranks = [torch.zeros_like(local_new_ranks) for _ in range(world_size)]
            dist.all_gather(gathered_ranks, local_new_ranks)
            
            # Combine results and add teleportation + dangling contributions
            new_ranks = torch.zeros(n, device='cuda:0')
            
            start_idx = 0
            for i, part_ranks in enumerate(gathered_ranks):
                part_size = part_ranks.size(0)
                new_ranks[start_idx:start_idx+part_size] = part_ranks
                start_idx += part_size
            
            # Add teleportation and dangling contributions
            teleport = (1 - damping) / n
            new_ranks += teleport + dangling_contribution
            
            # Check for convergence
            diff = torch.norm(new_ranks - old_ranks, 1).item()
            converged = diff < tol
            
            # Update ranks for next iteration
            ranks_global = new_ranks
            
            iterations += 1
            
            # Synchronize all processes
            dist.barrier()
        
        # Return results only from rank 0
        if rank == 0:
            return ranks_global, iterations
        else:
            return None, iterations
    
    @staticmethod
    def la_pagerank_multi_gpu_v2(adj_matrix_sparse, damping=0.85, max_iterations=100, tol=1e-6, num_gpus=None):
        """
        Multi-GPU PageRank implementation for single-node multi-GPU setup (simpler version)
        
        Parameters:
        -----------
        adj_matrix_sparse : torch.sparse.Tensor
            Sparse adjacency matrix representation of the graph
        damping : float
            Damping factor (default: 0.85)
        max_iterations : int
            Maximum number of iterations (default: 100)
        tol : float
            Convergence tolerance (default: 1e-6)
        num_gpus : int or None
            Number of GPUs to use. If None, uses all available GPUs.
                
        Returns:
        --------
        ranks : torch.Tensor
            Tensor of PageRank scores for each node
        iterations : int
            Number of iterations performed
        """
        # Determine number of GPUs to use
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = min(num_gpus, torch.cuda.device_count())
        
        if num_gpus <= 0:
            raise RuntimeError("No GPUs available for multi-GPU PageRank")
        
        if num_gpus == 1:
            # Fall back to single GPU version if only one GPU
            print("Using optimized single-GPU implementation for num_gpus=1")
            from src.algorithms.pagerank import PageRank
            return PageRank.la_pagerank_sparse_gpu_optimized_v2_turbo(adj_matrix_sparse, damping, max_iterations, tol)
        
        # Original matrix dimensions
        n = adj_matrix_sparse.shape[0]
        
        # Get graph size
        graph_size = n
        
        # For small graphs, fall back to single GPU version for better performance
        if graph_size <= 50000:  # Threshold based on testing
            print(f"Graph size {graph_size} <= 50000 nodes, using optimized single-GPU implementation")
            from src.algorithms.pagerank import PageRank
            return PageRank.la_pagerank_sparse_gpu_optimized_v2_turbo(adj_matrix_sparse, damping, max_iterations, tol)
        
        # For strong scaling, we'll use a different approach
        # Make a full copy of the matrix on each GPU for better performance
        # and more even load balancing
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        adj_matrices = []
        
        # For PageRank, we need the transposed matrix
        adj_matrix_t = adj_matrix_sparse.transpose(0, 1).coalesce()
        
        for i, device in enumerate(devices):
            # Move a copy of the transposed matrix to each GPU
            adj_matrices.append(adj_matrix_t.to(device))
        
        # Calculate out-degrees on main GPU (cuda:0)
        out_degrees = torch.sparse.sum(adj_matrix_sparse, dim=1).to_dense().to('cuda:0')
        
        # Compute inverse out-degrees and identify dangling nodes
        inv_out_degrees = torch.zeros_like(out_degrees)
        mask = out_degrees > 0
        inv_out_degrees[mask] = 1.0 / out_degrees[mask]
        dangling_mask = ~mask
        
        # Initialize PageRank vector on main GPU
        ranks = torch.full((n,), 1.0 / n, device='cuda:0')
        
        # For load balancing, assign chunks of the vector to different GPUs
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
        
        # Prepare for iteration
        iterations = 0
        converged = False
        
        # Main iteration loop
        while not converged and iterations < max_iterations:
            # Keep old ranks for convergence check
            old_ranks = ranks.clone()
            
            # Calculate dangling weight contribution
            dangling_sum = torch.sum(ranks[dangling_mask])
            dangling_contribution = damping * dangling_sum / n
            
            # Get chunks for distributing work
            chunks = get_node_chunks(n, num_gpus)
            
            # Process chunks on their respective GPUs
            partial_results = []
            for i, (start_idx, end_idx) in enumerate(chunks):
                # Get device
                device = devices[i]
                
                # Create a mask for this chunk
                chunk_size = end_idx - start_idx
                
                # Copy necessary data to this GPU
                ranks_gpu = ranks.to(device)
                inv_out_degrees_gpu = inv_out_degrees.to(device)
                
                # Apply inverse out-degrees
                scaled_ranks = ranks_gpu * inv_out_degrees_gpu
                
                # Each GPU processes its assigned range of the output
                partial_result = torch.zeros(n, device=device)
                
                # Matrix multiplication for assigned output range
                # Extract the part of the matrix corresponding to this chunk's output
                chunk_matrix = adj_matrices[i]
                
                # Compute contribution
                contribution = torch.sparse.mm(chunk_matrix, scaled_ranks.unsqueeze(1)).squeeze(1)
                
                # Only keep assigned range
                contribution_chunk = contribution[start_idx:end_idx]
                
                # Move result back to main GPU
                partial_results.append((start_idx, end_idx, contribution_chunk.to('cuda:0')))
            
            # Combine results from all GPUs
            new_ranks = torch.zeros(n, device='cuda:0')
            for start_idx, end_idx, contribution in partial_results:
                new_ranks[start_idx:end_idx] = contribution
            
            # Scale by damping factor
            new_ranks *= damping
            
            # Add teleportation and dangling contributions
            teleport = (1 - damping) / n
            new_ranks += teleport + dangling_contribution
            
            # Check for convergence
            diff = torch.norm(new_ranks - old_ranks, 1).item()
            converged = diff < tol
            
            # Update ranks for next iteration
            ranks = new_ranks
            
            iterations += 1
        
        return ranks, iterations