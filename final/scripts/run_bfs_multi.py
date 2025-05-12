#!/usr/bin/env python3
"""
Script to run BFS benchmarks comparing traditional and multiprocessing
implementations - without plotting or saving
"""

import os
import sys
import argparse
import numpy as np
import time
import multiprocessing

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
from src.utils.graph_utils import (
    generate_random_graph, 
    generate_scale_free_graph,
    generate_small_world_graph,
    graph_to_adj_list
)

def print_simplified_graph_stats(G):
    """Print only essential graph stats"""
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

def run_test(func, *args, n_runs=1, **kwargs):
    """Run a test function multiple times and return timing statistics"""
    times = []
    results = None
    
    for i in range(n_runs):
        start_time = time.time()
        results = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
    
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
    
    return result, results

def main():
    parser = argparse.ArgumentParser(description='Run BFS benchmarks for traditional and multiprocessing')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each benchmark')
    parser.add_argument('--processes', type=int, nargs='+', default=[2, 4, 8], 
                        help='Number of processes for multiprocessing implementations')
    parser.add_argument('--verify', action='store_true', help='Verify implementation correctness before benchmarking')
    parser.add_argument('--verify-size', type=int, default=500, help='Graph size for verification')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
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
    
    # Get max available cores
    max_cores = multiprocessing.cpu_count()
    print(f"System has {max_cores} CPU cores available")
    
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
        
        # Choose a start node (node with highest degree)
        start_node_verify = max(G_verify.degree(), key=lambda x: x[1])[0]
        
        verification_results = {}
        
        # Run traditional BFS
        print("Running traditional BFS for verification...")
        _, traditional_result = run_test(
            BFS.traditional_bfs_cpu,
            adj_list_verify,
            start_node_verify,
            n_runs=1
        )
        verification_results['Traditional_BFS'] = traditional_result
        
        # Run multiprocessing BFS with a reasonable process count
        process_count = min(4, max_cores)
        print(f"Running multiprocessing BFS with {process_count} processes for verification...")
        try:
            _, mp_result = run_test(
                BFS.traditional_bfs_multiprocessing,
                adj_list_verify,
                start_node_verify,
                process_count,
                n_runs=1
            )
            verification_results[f'MP_BFS_{process_count}processes'] = mp_result
        except Exception as e:
            print(f"Error in multiprocessing BFS verification: {e}")
        
        # Verify results
        print("\nVerifying implementation correctness...")
        all_correct = True
        reference_visited, reference_distances = verification_results['Traditional_BFS']
        
        for name, (visited, distances) in verification_results.items():
            if name == 'Traditional_BFS':
                continue
            
            # Check if all nodes in the reference implementation have the same distance
            distances_match = True
            for node, distance in reference_distances.items():
                if node not in distances or distances[node] != distance:
                    print(f"Distance mismatch in {name} for node {node}: "
                          f"Expected {distance}, Got {distances.get(node, 'missing')}")
                    distances_match = False
                    all_correct = False
                    break
            
            # Check if all nodes in this implementation are in the reference
            for node in distances:
                if node not in reference_distances:
                    print(f"Node {node} in {name} but not in reference implementation")
                    all_correct = False
                    break
            
            if distances_match:
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
        
        # Choose a start node (node with highest degree)
        start_node = max(G.degree(), key=lambda x: x[1])[0]
        
        # Store results for this size
        size_results = {}
        
        # Run traditional BFS
        print("\nRunning traditional BFS...")
        trad_result, _ = run_test(
            BFS.traditional_bfs_cpu,
            adj_list,
            start_node,
            n_runs=args.runs
        )
        size_results['Traditional_BFS'] = trad_result
        print(f"Average time: {trad_result['avg_time']:.6f} seconds")
        print(f"Std dev: {trad_result['std_time']:.6f} seconds")
        print(f"Min time: {trad_result['min_time']:.6f} seconds")
        print(f"Max time: {trad_result['max_time']:.6f} seconds")
        
        # Run multiprocessing-based BFS with different process counts
        print("\nRunning multiprocessing BFS implementation...")
        
        # Test with different process counts
        for process_count in args.processes:
            if process_count <= max_cores:
                print(f"  With {process_count} processes:")
                
                # Multiprocessing BFS
                try:
                    mp_result, _ = run_test(
                        BFS.traditional_bfs_multiprocessing,
                        adj_list,
                        start_node,
                        process_count,
                        n_runs=args.runs
                    )
                    size_results[f'MP_BFS_{process_count}processes'] = mp_result
                    print(f"  Average time: {mp_result['avg_time']:.6f} seconds")
                    print(f"  Std dev: {mp_result['std_time']:.6f} seconds")
                    print(f"  Min time: {mp_result['min_time']:.6f} seconds")
                    print(f"  Max time: {mp_result['max_time']:.6f} seconds")
                    print(f"  Speedup vs traditional: {trad_result['avg_time'] / mp_result['avg_time']:.2f}x")
                except Exception as e:
                    print(f"  Error in multiprocessing BFS with {process_count} processes: {e}")
        
        # Store results for this size
        all_results[size] = size_results
    
    # Print summary of all results
    print("\n" + "="*60)
    print(f"SUMMARY FOR {args.graph_type.upper()} GRAPHS")
    print("="*60)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)", "Speedup"]
    print(f"{headers[0]:<10} {headers[1]:<25} {headers[2]:<15} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15} {headers[6]:<10}")
    print("-" * 100)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        trad_time = size_results['Traditional_BFS']['avg_time']
        
        # Print traditional first
        alg_name = 'Traditional_BFS'
        result = size_results[alg_name]
        print(f"{size:<10} {alg_name:<25} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
              f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {1.0:<10.2f}")
        
        # Then print other algorithms
        for alg_name, result in size_results.items():
            if alg_name == 'Traditional_BFS':
                continue
            speedup = trad_time / result['avg_time']
            print(f"{'':<10} {alg_name:<25} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
                  f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {speedup:<10.2f}")
        
        print("-" * 100)

if __name__ == '__main__':
    main()