#!/usr/bin/env python3
"""
Script to run BFS benchmarks comparing traditional and linear algebra implementations
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from src.utils.benchmark import Benchmark

def main():
    parser = argparse.ArgumentParser(description='Run BFS benchmarks')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 5000, 10000, 15000, 20000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for each benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
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
    
    # Create benchmark object
    benchmark = Benchmark(f'bfs_{args.graph_type}', save_dir=args.save_dir)
    
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
    
    # Run benchmarks for each graph size
    for size in args.sizes:
        print(f"\nBenchmarking graph of size {size}...")
        
        # Generate graph
        G = generate_graph(size)
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
        
        # Run traditional BFS
        print("Running traditional BFS...")
        trad_result = benchmark.run_test(
            BFS.traditional_bfs_cpu,
            adj_list,
            start_node,
            n_runs=args.runs
        )
        benchmark.add_result('Traditional_BFS', args.graph_type, size, trad_result)
        print(f"Average time: {trad_result['avg_time']:.6f} seconds")
        
        # Run linear algebra BFS on CPU
        print("Running linear algebra BFS on CPU...")
        la_cpu_result = benchmark.run_test(
            BFS.la_bfs_cpu,
            adj_matrix_np,
            start_node,
            n_runs=args.runs
        )
        benchmark.add_result('LA_BFS_CPU', args.graph_type, size, la_cpu_result)
        print(f"Average time: {la_cpu_result['avg_time']:.6f} seconds")
        
        # Run linear algebra BFS on GPU if requested
        if args.gpu:
            print("Running linear algebra BFS on GPU (dense)...")
            la_gpu_result = benchmark.run_test(
                BFS.la_bfs_gpu,
                adj_matrix_torch,
                start_node,
                n_runs=args.runs
            )
            benchmark.add_result('LA_BFS_GPU_Dense', args.graph_type, size, la_gpu_result)
            print(f"Average time: {la_gpu_result['avg_time']:.6f} seconds")
            
            print("Running linear algebra BFS on GPU (sparse)...")
            la_sparse_result = benchmark.run_test(
                BFS.la_bfs_sparse_gpu,
                adj_matrix_sparse,
                start_node,
                n_runs=args.runs
            )
            benchmark.add_result('LA_BFS_GPU_Sparse', args.graph_type, size, la_sparse_result)
            print(f"Average time: {la_sparse_result['avg_time']:.6f} seconds")
    
    # Save results
    benchmark.save_results()
    benchmark.print_results()
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating performance comparison plot...")
        benchmark.plot_comparison(
            args.graph_type,
            save_file=f'bfs_{args.graph_type}_performance.png'
        )
        
        print("Generating speedup comparison plot...")
        benchmark.plot_speedup(
            'Traditional_BFS',
            args.graph_type,
            save_file=f'bfs_{args.graph_type}_speedup.png'
        )

if __name__ == '__main__':
    main()