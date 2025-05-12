#!/usr/bin/env python3
"""
Script to verify correctness of PageRank implementations
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

from src.algorithms.pagerank import PageRank
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
from src.utils.verification_omp import verify_pagerank_correctness, print_algorithm_stats

def main():
    parser = argparse.ArgumentParser(description='Verify correctness of PageRank implementations')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Graph size for verification')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor for PageRank')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations for PageRank')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance for PageRank')
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
    
    # Run all PageRank implementations
    pagerank_results = {}
    
    print("\nRunning PageRank implementations...")
    
    # Traditional PageRank (baseline)
    print("Running traditional PageRank...")
    traditional_result = PageRank.traditional_pagerank_cpu(
        adj_list, 
        damping=args.damping, 
        max_iterations=args.max_iters, 
        tol=args.tolerance
    )
    pagerank_results['Traditional_PageRank'] = traditional_result
    
    # OpenMP PageRank with different thread counts
    print("Running OpenMP PageRank implementations...")
    for thread_count in threads_to_test:
        print(f"  With {thread_count} threads:")
        openmp_result = PageRank.traditional_pagerank_openmp(
            adj_list,
            damping=args.damping,
            max_iterations=args.max_iters,
            tol=args.tolerance,
            num_threads=thread_count
        )
        pagerank_results[f'OpenMP_PageRank_{thread_count}threads'] = openmp_result
    
    # Linear algebra PageRank on CPU
    print("Running linear algebra PageRank on CPU...")
    la_cpu_result = PageRank.la_pagerank_cpu(
        adj_matrix_np, 
        damping=args.damping, 
        max_iterations=args.max_iters, 
        tol=args.tolerance
    )
    pagerank_results['LA_PageRank_CPU'] = la_cpu_result
    
    # Linear algebra PageRank on GPU if requested
    if args.gpu:
        print("Running linear algebra PageRank on GPU (dense)...")
        la_gpu_result = PageRank.la_pagerank_gpu(
            adj_matrix_torch, 
            damping=args.damping, 
            max_iterations=args.max_iters, 
            tol=args.tolerance
        )
        pagerank_results['LA_PageRank_GPU_Dense'] = la_gpu_result
        
        print("Running linear algebra PageRank on GPU (sparse)...")
        la_sparse_result = PageRank.la_pagerank_sparse_gpu(
            adj_matrix_sparse, 
            damping=args.damping, 
            max_iterations=args.max_iters, 
            tol=args.tolerance
        )
        pagerank_results['LA_PageRank_GPU_Sparse'] = la_sparse_result
    
    # Verify correctness
    print("\nVerifying correctness...")
    is_correct = verify_pagerank_correctness(pagerank_results, tolerance=1e-5)
    print(f"PageRank implementations correctness check: {'PASSED' if is_correct else 'FAILED'}")
    
    # Print detailed statistics if requested
    if args.verbose:
        print_algorithm_stats(pagerank_results)
    
    # Compare with NetworkX implementation
    print("\nComparing with NetworkX implementation...")
    nx_pagerank = nx.pagerank(G, alpha=args.damping, tol=args.tolerance)
    
    # Convert to the same format as our implementation
    nx_result = (nx_pagerank, 0)  # NetworkX doesn't return iteration count
    
    # Create a new dictionary with our implementations and NetworkX
    compare_results = {'NetworkX': nx_result}
    compare_results.update(pagerank_results)
    
    # Verify correctness against NetworkX
    is_nx_correct = verify_pagerank_correctness(compare_results, tolerance=1e-4)
    print(f"NetworkX comparison correctness check: {'PASSED' if is_nx_correct else 'FAILED'}")
    
    # Analyze differences if verbose
    if args.verbose and is_nx_correct:
        # Compare iteration counts
        for algo, (ranks, iterations) in pagerank_results.items():
            if iterations > 0:  # Skip NetworkX which doesn't report iterations
                print(f"{algo}: {iterations} iterations to converge")
        
        # Compare rank values for a few random nodes
        sample_nodes = np.random.choice(list(nx_pagerank.keys()), min(5, len(nx_pagerank)), replace=False)
        print("\nSample node PageRank values:")
        print(f"{'Node':<10} {'NetworkX':<15} {'Traditional':<15} {'OpenMP':<15} {'LA_CPU':<15}")
        for node in sample_nodes:
            trad_val = pagerank_results['Traditional_PageRank'][0][node]
            openmp_key = f'OpenMP_PageRank_{threads_to_test[-1]}threads'
            openmp_val = pagerank_results[openmp_key][0][node]
            la_cpu_val = pagerank_results['LA_PageRank_CPU'][0][node]
            print(f"{node:<10} {nx_pagerank[node]:<15.8f} {trad_val:<15.8f} {openmp_val:<15.8f} {la_cpu_val:<15.8f}")

if __name__ == '__main__':
    main()