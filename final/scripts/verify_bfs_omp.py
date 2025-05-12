#!/usr/bin/env python3
"""
Script to verify correctness of BFS implementations
"""

import os
import sys
import argparse
import torch
import numpy as np
import networkx as nx
import multiprocessing

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
from src.utils.graph_utils import (
    generate_random_graph, 
    generate_scale_free_graph,
    generate_small_world_graph,
    graph_to_adj_list,
    graph_to_adj_matrix_numpy,
    graph_to_adj_matrix_torch,
    graph_to_sparse_adj_matrix_torch,
    print_graph_stats
)
from src.utils.verification_omp import verify_bfs_correctness, print_algorithm_stats

def main():
    parser = argparse.ArgumentParser(description='Verify correctness of BFS implementations')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Graph size for verification')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--verbose', action='store_true', help='Print detailed statistics')
    parser.add_argument('--threads', type=int, nargs='+', default=[2, 4, 8], 
                        help='Thread counts to test for OpenMP implementations')
    parser.add_argument('--max-threads', type=int, default=32,
                        help='Maximum number of threads to use, even if more cores are available')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check if GPU is available if requested
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
        args.gpu = False
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Define graph generation function based on type
    if args.graph_type == 'random':
        def generate_graph(n):
            # Adjust p to maintain average degree around 10
            p = 10 / (n - 1)
            return generate_random_graph(n, p, seed=args.seed)
    elif args.graph_type == 'scale-free':
        def generate_graph(n):
            # Each new node adds 5 edges, so average degree is around 10
            return generate_scale_free_graph(n, 5, seed=args.seed)
    elif args.graph_type == 'small-world':
        def generate_graph(n):
            # Each node connected to 10 nearest neighbors (5 on each side)
            # Rewiring probability of 0.1
            return generate_small_world_graph(n, 10, 0.1, seed=args.seed)
    
    # Get available threads, capped at max-threads
    available_cores = min(multiprocessing.cpu_count(), args.max_threads)
    print(f"System has {multiprocessing.cpu_count()} CPU cores available, using maximum of {available_cores}")
    
    # Use only the thread counts specified, don't add the max automatically
    threads_to_test = [t for t in args.threads if t <= available_cores]
    print(f"Testing with thread counts: {threads_to_test}")
    
    print(f"\nGenerating {args.graph_type} graph of size {args.size}...")
    
    # Generate graph
    G = generate_graph(args.size)
    print_graph_stats(G)
    
    # Convert graph to different representations
    adj_list = graph_to_adj_list(G)
    adj_matrix_np = graph_to_adj_matrix_numpy(G)
    
    if args.gpu:
        adj_matrix_torch = graph_to_adj_matrix_torch(G, device=device)
        adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
    
    # Choose a start node (node with highest degree)
    start_node = max(G.degree(), key=lambda x: x[1])[0]
    print(f"Using start node {start_node} with degree {G.degree(start_node)}")
    
    # Run all BFS implementations
    bfs_results = {}
    
    print("\nRunning BFS implementations...")
    
    # Traditional BFS (baseline)
    print("Running traditional BFS...")
    traditional_result = BFS.traditional_bfs_cpu(adj_list, start_node)
    bfs_results['Traditional_BFS'] = traditional_result
    
    # OpenMP BFS with different thread counts
    print("Running OpenMP BFS implementations...")
    for thread_count in threads_to_test:
        print(f"  With {thread_count} threads:")
        openmp_result = BFS.traditional_bfs_openmp(adj_list, start_node, thread_count)
        bfs_results[f'OpenMP_BFS_{thread_count}threads'] = openmp_result
    
    # Linear algebra BFS on CPU
    print("Running linear algebra BFS on CPU...")
    la_cpu_result = BFS.la_bfs_cpu(adj_matrix_np, start_node)
    bfs_results['LA_BFS_CPU'] = la_cpu_result
    
    # Linear algebra BFS on GPU if requested
    if args.gpu:
        print("Running linear algebra BFS on GPU (dense)...")
        la_gpu_result = BFS.la_bfs_gpu(adj_matrix_torch, start_node)
        bfs_results['LA_BFS_GPU_Dense'] = la_gpu_result
        
        print("Running linear algebra BFS on GPU (sparse)...")
        la_sparse_result = BFS.la_bfs_sparse_gpu(adj_matrix_sparse, start_node)
        bfs_results['LA_BFS_GPU_Sparse'] = la_sparse_result
    
    # Verify correctness
    print("\nVerifying correctness...")
    is_correct = verify_bfs_correctness(bfs_results)
    print(f"BFS implementations correctness check: {'PASSED' if is_correct else 'FAILED'}")
    
    # Print detailed statistics if requested
    if args.verbose:
        print_algorithm_stats(bfs_results)
    
    # Compare with NetworkX implementation
    print("\nComparing with NetworkX implementation...")
    nx_distances = nx.single_source_shortest_path_length(G, start_node)
    
    # Convert visited and distances to match our format
    nx_visited = list(nx.bfs_tree(G, start_node))
    
    # Create a result tuple like ours
    nx_result = (nx_visited, nx_distances)
    
    # Create a new dictionary with our implementations and NetworkX
    compare_results = {'NetworkX': nx_result}
    compare_results.update(bfs_results)
    
    # Verify correctness against NetworkX
    is_nx_correct = verify_bfs_correctness(compare_results)
    print(f"NetworkX comparison correctness check: {'PASSED' if is_nx_correct else 'FAILED'}")

if __name__ == '__main__':
    main()