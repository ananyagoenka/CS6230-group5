#!/usr/bin/env python3
"""
Script to run PageRank benchmarks comparing only traditional CPU and sparse GPU
implementations - without plotting or saving
"""

import os
import sys
import argparse
import torch
import numpy as np
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.pagerank import PageRank
from src.utils.graph_utils import (
    generate_random_graph, 
    generate_scale_free_graph,
    generate_small_world_graph,
    graph_to_adj_list,
    graph_to_sparse_adj_matrix_torch
)

def print_simplified_graph_stats(G):
    """Print only essential graph stats"""
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

def run_test(func, *args, n_runs=1, **kwargs):
    """Run a test function multiple times and return timing statistics"""
    times = []
    iter_counts = []
    results = None
    
    for i in range(n_runs):
        start_time = time.time()
        results = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        
        # For PageRank, capture iteration counts
        if isinstance(results, tuple) and len(results) == 2 and isinstance(results[1], int):
            ranks, iterations = results
            iter_counts.append(iterations)
    
    avg_time = sum(times) / n_runs
    min_time = min(times)
    max_time = max(times)
    std_time = np.std(times) if n_runs > 1 else 0.0
    
    result = {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time
    }
    
    # Add iterations info if available
    if iter_counts:
        result['iterations'] = sum(iter_counts) / len(iter_counts)
    
    return result, results

def main():
    parser = argparse.ArgumentParser(description='Run PageRank benchmarks (traditional CPU and sparse GPU only)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each benchmark')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor for PageRank')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations for PageRank')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance for PageRank')
    parser.add_argument('--verify', action='store_true', help='Verify implementation correctness before benchmarking')
    parser.add_argument('--verify-size', type=int, default=500, help='Graph size for verification')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("Error: GPU not available. This script requires GPU support.")
        return
    
    device = torch.device('cuda')
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
    
    # Store results for summary
    all_results = {}
    
    # Verify implementation correctness if requested
    if args.verify:
        print(f"\nVerifying implementation correctness with graph of size {args.verify_size}...")
        
        # Generate verification graph
        G_verify = generate_graph(args.verify_size)
        print_simplified_graph_stats(G_verify)
        
        # Convert graph to different representations
        adj_list_verify = graph_to_adj_list(G_verify)
        adj_matrix_sparse_verify = graph_to_sparse_adj_matrix_torch(G_verify, device=device)
        
        verification_results = {}
        
        # Run traditional PageRank
        print("Running traditional PageRank for verification...")
        _, traditional_result = run_test(
            PageRank.traditional_pagerank_cpu,
            adj_list_verify, 
            damping=args.damping, 
            max_iterations=args.max_iters, 
            tol=args.tolerance,
            n_runs=1
        )
        verification_results['Traditional_PageRank'] = traditional_result
        
        # Run sparse GPU PageRank
        print("Running linear algebra PageRank on GPU (sparse) for verification...")
        try:
            _, la_sparse_result = run_test(
                PageRank.la_pagerank_sparse_gpu,
                adj_matrix_sparse_verify,
                damping=args.damping,
                max_iterations=args.max_iters,
                tol=args.tolerance,
                n_runs=1
            )
            verification_results['LA_PageRank_GPU_Sparse'] = la_sparse_result
        except Exception as e:
            print(f"Error in LA GPU Sparse PageRank verification: {e}")
        
        # Verify results
        print("\nVerifying implementation correctness...")
        all_correct = True
        reference_ranks, _ = verification_results['Traditional_PageRank']
        
        for name, (ranks, _) in verification_results.items():
            if name == 'Traditional_PageRank':
                continue
            
            # Check if all nodes in the reference have similar rank
            ranks_match = True
            for node, rank in reference_ranks.items():
                if node not in ranks:
                    print(f"Node {node} missing in {name}")
                    ranks_match = False
                    all_correct = False
                    break
                
                # Check if the ranks are similar within tolerance
                if abs(float(rank) - float(ranks[node])) > 1e-5:
                    print(f"Rank mismatch in {name} for node {node}: "
                          f"Expected {rank}, Got {ranks[node]}, Diff {abs(float(rank) - float(ranks[node]))}")
                    ranks_match = False
                    all_correct = False
                    break
            
            # Check if all nodes in this implementation are in the reference
            for node in ranks:
                if node not in reference_ranks:
                    print(f"Node {node} in {name} but not in reference implementation")
                    ranks_match = False
                    all_correct = False
                    break
            
            if ranks_match:
                print(f"{name} matches the reference implementation")
        
        if not all_correct:
            print("\nWARNING: Some implementations do not match the reference!")
            proceed = input("Do you want to proceed with benchmarking anyway? (y/n): ")
            if proceed.lower() != 'y':
                print("Exiting...")
                return
    
    # Run benchmarks for each graph size
    for size in args.sizes:
        print(f"\n{'-'*40}")
        print(f"Benchmarking graph of size {size}...")
        print(f"{'-'*40}")
        
        # Generate graph
        G = generate_graph(size)
        print_simplified_graph_stats(G)
        
        # Convert graph to different representations
        adj_list = graph_to_adj_list(G)
        adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
        
        # Store results for this size
        size_results = {}
        
        # Run traditional PageRank
        print("\nRunning traditional PageRank...")
        try:
            trad_result, _ = run_test(
                PageRank.traditional_pagerank_cpu,
                adj_list,
                damping=args.damping,
                max_iterations=args.max_iters,
                tol=args.tolerance,
                n_runs=args.runs
            )
            size_results['Traditional_PageRank'] = trad_result
            print(f"Average time: {trad_result['avg_time']:.6f} seconds")
            print(f"Std dev: {trad_result['std_time']:.6f} seconds")
            print(f"Min time: {trad_result['min_time']:.6f} seconds")
            print(f"Max time: {trad_result['max_time']:.6f} seconds")
            if 'iterations' in trad_result:
                print(f"Iterations: {trad_result['iterations']:.1f}")
        except Exception as e:
            print(f"Error in traditional PageRank: {e}")
        
        # Run sparse GPU PageRank
        print("\nRunning linear algebra PageRank on GPU (sparse)...")
        try:
            la_sparse_result, _ = run_test(
                PageRank.la_pagerank_sparse_gpu,
                adj_matrix_sparse,
                damping=args.damping,
                max_iterations=args.max_iters,
                tol=args.tolerance,
                n_runs=args.runs
            )
            size_results['LA_PageRank_GPU_Sparse'] = la_sparse_result
            print(f"Average time: {la_sparse_result['avg_time']:.6f} seconds")
            print(f"Std dev: {la_sparse_result['std_time']:.6f} seconds")
            print(f"Min time: {la_sparse_result['min_time']:.6f} seconds")
            print(f"Max time: {la_sparse_result['max_time']:.6f} seconds")
            if 'iterations' in la_sparse_result:
                print(f"Iterations: {la_sparse_result['iterations']:.1f}")
            print(f"Speedup vs traditional: {trad_result['avg_time'] / la_sparse_result['avg_time']:.2f}x")
        except Exception as e:
            print(f"Error in LA PageRank GPU (sparse): {e}")
                
        # Store results for this size
        all_results[size] = size_results
    
    # Print summary of all results
    print("\n" + "="*60)
    print(f"SUMMARY FOR {args.graph_type.upper()} GRAPHS")
    print("="*60)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)"]
    if any('iterations' in result for size_results in all_results.values() for result in size_results.values()):
        headers.append("Iterations")
    headers.append("Speedup")
    
    header_str = f"{headers[0]:<10} {headers[1]:<25} {headers[2]:<15} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15}"
    if len(headers) > 6:
        header_str += f" {headers[6]:<10}"
    if len(headers) > 7:
        header_str += f" {headers[7]:<10}"
    print(header_str)
    print("-" * 100)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        if 'Traditional_PageRank' not in size_results:
            continue
            
        trad_time = size_results['Traditional_PageRank']['avg_time']
        
        # Print traditional first
        alg_name = 'Traditional_PageRank'
        result = size_results[alg_name]
        
        result_str = f"{size:<10} {alg_name:<25} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
        result_str += f"{result['min_time']:<15.6f} {result['max_time']:<15.6f}"
        if 'iterations' in result:
            result_str += f" {result['iterations']:<10.1f}"
        elif 'Iterations' in headers:
            result_str += f" {'':<10}"
        result_str += f" {1.0:<10.2f}"
        print(result_str)
        
        # Print sparse GPU
        alg_name = 'LA_PageRank_GPU_Sparse'
        if alg_name in size_results:
            result = size_results[alg_name]
            speedup = trad_time / result['avg_time']
            
            result_str = f"{'':<10} {alg_name:<25} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
            result_str += f"{result['min_time']:<15.6f} {result['max_time']:<15.6f}"
            if 'iterations' in result:
                result_str += f" {result['iterations']:<10.1f}"
            elif 'Iterations' in headers:
                result_str += f" {'':<10}"
            result_str += f" {speedup:<10.2f}"
            print(result_str)
        
        print("-" * 100)

if __name__ == '__main__':
    main()