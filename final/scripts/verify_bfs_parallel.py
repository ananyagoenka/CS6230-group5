#!/usr/bin/env python3
"""
Script to verify correctness of BFS implementations including multiprocessing and Numba
"""

import os
import sys
import argparse
import torch
import numpy as np
import networkx as nx
import multiprocessing
import time
from collections import defaultdict

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

def verify_bfs_correctness(bfs_results):
    """
    Verify that all BFS implementations produce the same results
    
    Parameters:
    -----------
    bfs_results : dict
        Dictionary mapping implementation names to BFS results (visited, distances)
        
    Returns:
    --------
    is_correct : bool
        True if all implementations produce identical results
    """
    if not bfs_results:
        return True
    
    # Get reference implementation (first in the dictionary)
    reference_name = list(bfs_results.keys())[0]
    reference_visited, reference_distances = bfs_results[reference_name]
    
    all_correct = True
    
    for name, (visited, distances) in bfs_results.items():
        if name == reference_name:
            continue
        
        # Check if all nodes in the reference implementation have the same distance
        distances_match = True
        for node, distance in reference_distances.items():
            if node not in distances or distances[node] != distance:
                print(f"Mismatch in {name} for node {node}: Expected {distance}, Got {distances.get(node, 'missing')}")
                distances_match = False
                all_correct = False
                break
        
        # Check if all nodes in this implementation are in the reference
        nodes_match = True
        for node in distances:
            if node not in reference_distances:
                print(f"Node {node} in {name} but not in reference implementation")
                nodes_match = False
                all_correct = False
                break
        
        if distances_match and nodes_match:
            print(f"{name} matches the reference implementation")
    
    return all_correct

def print_algorithm_stats(results, time_results=None):
    """
    Print detailed statistics for algorithm results
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping implementation names to algorithm results
    time_results : dict or None
        Dictionary mapping implementation names to execution times
    """
    print("\nAlgorithm Statistics:")
    print("=" * 80)
    
    if not results:
        print("No results to display.")
        return
    
    # BFS specific statistics
    if isinstance(list(results.values())[0][0], list):  # First element is a list of visited nodes
        max_distances = {}
        visited_counts = {}
        for name, (visited, distances) in results.items():
            max_dist = max(d for d in distances.values() if d != float('infinity'))
            max_distances[name] = max_dist
            visited_counts[name] = len(visited)
        
        # Print header with timing info if available
        if time_results:
            print(f"{'Implementation':<30} {'Max Distance':<15} {'Visited Nodes':<15} {'Time (s)':<15}")
        else:
            print(f"{'Implementation':<30} {'Max Distance':<15} {'Visited Nodes':<15}")
        print("-" * 80)
        
        # Print stats for each implementation
        for name in results:
            if time_results:
                time_taken = time_results.get(name, "N/A")
                if isinstance(time_taken, (int, float)):
                    time_str = f"{time_taken:.6f}"
                else:
                    time_str = time_taken
                print(f"{name:<30} {max_distances[name]:<15} {visited_counts[name]:<15} {time_str:<15}")
            else:
                print(f"{name:<30} {max_distances[name]:<15} {visited_counts[name]:<15}")
        
        # Calculate speedup relative to sequential if timing info is available
        if time_results and "Traditional_BFS" in time_results:
            baseline_time = time_results["Traditional_BFS"]
            if isinstance(baseline_time, (int, float)) and baseline_time > 0:
                print("\nSpeedup relative to sequential implementation:")
                for name, time_taken in time_results.items():
                    if name != "Traditional_BFS" and isinstance(time_taken, (int, float)):
                        speedup = baseline_time / time_taken
                        print(f"{name:<30} {speedup:.2f}x")

def main():
    parser = argparse.ArgumentParser(description='Verify correctness of BFS implementations')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Graph size for verification')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print detailed statistics')
    parser.add_argument('--processes', type=int, nargs='+', default=[2, 4, 8], 
                        help='Process counts to test for multiprocessing implementations')
    parser.add_argument('--max-processes', type=int, default=32,
                        help='Maximum number of processes to use, even if more cores are available')
    parser.add_argument('--timing', action='store_true', help='Measure execution time')
    parser.add_argument('--repeat', type=int, default=1, help='Number of timing repetitions')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get available cores, capped at max-processes
    available_cores = min(multiprocessing.cpu_count(), args.max_processes)
    print(f"System has {multiprocessing.cpu_count()} CPU cores available, using maximum of {available_cores}")
    
    # Use only the process counts specified, don't add the max automatically
    processes_to_test = [p for p in args.processes if p <= available_cores]
    print(f"Testing with process counts: {processes_to_test}")
    
    print(f"\nGenerating {args.graph_type} graph of size {args.size}...")
    
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
    
    # Generate graph
    G = generate_graph(args.size)
    print_graph_stats(G)
    
    # Convert graph to different representations
    adj_list = graph_to_adj_list(G)
    adj_matrix_np = graph_to_adj_matrix_numpy(G)
    
    # Choose a start node (node with highest degree)
    start_node = max(G.degree(), key=lambda x: x[1])[0]
    print(f"Using start node {start_node} with degree {G.degree(start_node)}")
    
    # Run all BFS implementations
    bfs_results = {}
    time_results = {} if args.timing else None
    
    print("\nRunning BFS implementations...")
    
    # Traditional BFS (baseline)
    print("Running traditional BFS...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            traditional_result = BFS.traditional_bfs_cpu(adj_list, start_node)
        time_results["Traditional_BFS"] = (time.time() - start_time) / args.repeat
    else:
        traditional_result = BFS.traditional_bfs_cpu(adj_list, start_node)
    bfs_results['Traditional_BFS'] = traditional_result
    
    # Multiprocessing BFS with different process counts
    print("Running Multiprocessing BFS implementations...")
    for process_count in processes_to_test:
        print(f"  With {process_count} processes:")
        if args.timing:
            start_time = time.time()
            for _ in range(args.repeat):
                mp_result = BFS.traditional_bfs_multiprocessing(adj_list, start_node, process_count)
            time_results[f'Multiprocessing_BFS_{process_count}processes'] = (time.time() - start_time) / args.repeat
        else:
            mp_result = BFS.traditional_bfs_multiprocessing(adj_list, start_node, process_count)
        bfs_results[f'Multiprocessing_BFS_{process_count}processes'] = mp_result
    
    # Verify correctness
    print("\nVerifying correctness...")
    is_correct = verify_bfs_correctness(bfs_results)
    print(f"BFS implementations correctness check: {'PASSED' if is_correct else 'FAILED'}")
    
    # Print detailed statistics if requested
    if args.verbose:
        print_algorithm_stats(bfs_results, time_results)
    
    # Compare with NetworkX implementation
    print("\nComparing with NetworkX implementation...")
    
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            nx_distances = nx.single_source_shortest_path_length(G, start_node)
            nx_visited = list(nx.bfs_tree(G, start_node))
        time_results['NetworkX'] = (time.time() - start_time) / args.repeat
    else:
        nx_distances = nx.single_source_shortest_path_length(G, start_node)
        nx_visited = list(nx.bfs_tree(G, start_node))
    
    # Create a result tuple like ours
    nx_result = (nx_visited, nx_distances)
    
    # Create a new dictionary with our implementations and NetworkX
    compare_results = {'NetworkX': nx_result}
    compare_results.update(bfs_results)
    
    # Verify correctness against NetworkX
    is_nx_correct = verify_bfs_correctness(compare_results)
    print(f"NetworkX comparison correctness check: {'PASSED' if is_nx_correct else 'FAILED'}")
    
    # Print timing summary if requested
    if args.timing and args.verbose:
        print("\nTiming Summary:")
        print("=" * 80)
        print(f"{'Implementation':<30} {'Average Time (s)':<20}")
        print("-" * 80)
        for name, time_taken in time_results.items():
            print(f"{name:<30} {time_taken:.6f}")

if __name__ == '__main__':
    main()