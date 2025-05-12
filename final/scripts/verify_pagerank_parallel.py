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
import time

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

def verify_pagerank_correctness(pagerank_results, tolerance=1e-5):
    """
    Verify that all PageRank implementations produce similar results
    
    Parameters:
    -----------
    pagerank_results : dict
        Dictionary mapping implementation names to PageRank results (ranks, iterations)
    tolerance : float
        Maximum allowed difference in PageRank values
        
    Returns:
    --------
    is_correct : bool
        True if all implementations produce similar results within tolerance
    """
    if not pagerank_results:
        return True
    
    # Get reference implementation (first in the dictionary)
    reference_name = list(pagerank_results.keys())[0]
    reference_ranks, _ = pagerank_results[reference_name]
    
    all_correct = True
    
    for name, (ranks, _) in pagerank_results.items():
        if name == reference_name:
            continue
        
        # Check if all nodes in the reference implementation have similar rank
        ranks_match = True
        for node, rank in reference_ranks.items():
            if node not in ranks:
                print(f"Node {node} missing in {name}")
                ranks_match = False
                all_correct = False
                break
            
            # Convert to float if needed (to handle numpy/torch types)
            ref_rank = float(rank)
            test_rank = float(ranks[node])
            
            # Check if the ranks are similar within tolerance
            if abs(ref_rank - test_rank) > tolerance:
                print(f"Rank mismatch in {name} for node {node}: "
                      f"Expected {ref_rank}, Got {test_rank}, Diff {abs(ref_rank - test_rank)}")
                ranks_match = False
                all_correct = False
                break
        
        # Check if all nodes in this implementation are in the reference
        nodes_match = True
        for node in ranks:
            if node not in reference_ranks:
                print(f"Node {node} in {name} but not in reference implementation")
                nodes_match = False
                all_correct = False
                break
        
        if ranks_match and nodes_match:
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
    print("\nPageRank Algorithm Statistics:")
    print("=" * 80)
    
    if not results:
        print("No results to display.")
        return
    
    # PageRank specific statistics
    iterations = {}
    min_ranks = {}
    max_ranks = {}
    avg_ranks = {}
    
    for name, (ranks, iters) in results.items():
        iterations[name] = iters
        rank_values = [float(r) for r in ranks.values()]
        min_ranks[name] = min(rank_values)
        max_ranks[name] = max(rank_values)
        avg_ranks[name] = sum(rank_values) / len(rank_values)
    
    # Print header with timing info if available
    if time_results:
        print(f"{'Implementation':<30} {'Iterations':<10} {'Min Rank':<15} {'Max Rank':<15} {'Avg Rank':<15} {'Time (s)':<15}")
    else:
        print(f"{'Implementation':<30} {'Iterations':<10} {'Min Rank':<15} {'Max Rank':<15} {'Avg Rank':<15}")
    print("-" * 90)
    
    # Print stats for each implementation
    for name in results:
        if time_results:
            time_taken = time_results.get(name, "N/A")
            if isinstance(time_taken, (int, float)):
                time_str = f"{time_taken:.6f}"
            else:
                time_str = time_taken
            print(f"{name:<30} {iterations[name]:<10} {min_ranks[name]:<15.8f} {max_ranks[name]:<15.8f} {avg_ranks[name]:<15.8f} {time_str:<15}")
        else:
            print(f"{name:<30} {iterations[name]:<10} {min_ranks[name]:<15.8f} {max_ranks[name]:<15.8f} {avg_ranks[name]:<15.8f}")
    
    # Calculate speedup relative to sequential if timing info is available
    if time_results and "Traditional_PageRank" in time_results:
        baseline_time = time_results["Traditional_PageRank"]
        if isinstance(baseline_time, (int, float)) and baseline_time > 0:
            print("\nSpeedup relative to sequential implementation:")
            for name, time_taken in time_results.items():
                if name != "Traditional_PageRank" and isinstance(time_taken, (int, float)):
                    speedup = baseline_time / time_taken
                    print(f"{name:<30} {speedup:.2f}x")

def main():
    parser = argparse.ArgumentParser(description='Verify correctness of PageRank implementations')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Graph size for verification')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor for PageRank')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations for PageRank')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance for PageRank')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
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
    
    # Check if GPU is available if requested
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
        args.gpu = False
    
    device = torch.device('cuda' if args.gpu else 'cpu')
    print(f"Using device: {device}")
    
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
    
    if args.gpu:
        adj_matrix_torch = graph_to_adj_matrix_torch(G, device=device)
        adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
    
    # Run all PageRank implementations
    pagerank_results = {}
    time_results = {} if args.timing else None
    
    print("\nRunning PageRank implementations...")
    
    # Traditional PageRank (baseline)
    print("Running traditional PageRank...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            traditional_result = PageRank.traditional_pagerank_cpu(
                adj_list, 
                damping=args.damping, 
                max_iterations=args.max_iters, 
                tol=args.tolerance
            )
        time_results["Traditional_PageRank"] = (time.time() - start_time) / args.repeat
    else:
        traditional_result = PageRank.traditional_pagerank_cpu(
            adj_list, 
            damping=args.damping, 
            max_iterations=args.max_iters, 
            tol=args.tolerance
        )
    pagerank_results['Traditional_PageRank'] = traditional_result
    print(f"Iterations: {traditional_result[1]}")
    
    # Multiprocessing PageRank with different process counts
    print("Running Multiprocessing PageRank implementations...")
    for process_count in processes_to_test:
        print(f"  With {process_count} processes:")
        try:
            if args.timing:
                start_time = time.time()
                for _ in range(args.repeat):
                    mp_result = PageRank.traditional_pagerank_multiprocessing(
                        adj_list,
                        damping=args.damping,
                        max_iterations=args.max_iters,
                        tol=args.tolerance,
                        num_processes=process_count
                    )
                time_results[f'MP_PageRank_{process_count}processes'] = (time.time() - start_time) / args.repeat
            else:
                mp_result = PageRank.traditional_pagerank_multiprocessing(
                    adj_list,
                    damping=args.damping,
                    max_iterations=args.max_iters,
                    tol=args.tolerance,
                    num_processes=process_count
                )
            pagerank_results[f'MP_PageRank_{process_count}processes'] = mp_result
            print(f"  Iterations: {mp_result[1]}")
        except Exception as e:
            print(f"  Error in multiprocessing PageRank with {process_count} processes: {e}")
    

    
    # Verify correctness
    print("\nVerifying correctness...")
    is_correct = verify_pagerank_correctness(pagerank_results, tolerance=1e-5)
    print(f"PageRank implementations correctness check: {'PASSED' if is_correct else 'FAILED'}")
    
    # Print detailed statistics if requested
    if args.verbose:
        print_algorithm_stats(pagerank_results, time_results)
    
    # Compare with NetworkX implementation
    print("\nComparing with NetworkX implementation...")
    
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            nx_pagerank = nx.pagerank(G, alpha=args.damping, tol=args.tolerance)
        time_results['NetworkX'] = (time.time() - start_time) / args.repeat
    else:
        nx_pagerank = nx.pagerank(G, alpha=args.damping, tol=args.tolerance)
    
    # Convert to the same format as our implementation
    nx_result = (nx_pagerank, 0)  # NetworkX doesn't return iteration count
    
    # Create a new dictionary with our implementations and NetworkX
    compare_results = {'NetworkX': nx_result}
    compare_results.update(pagerank_results)
    
    # Verify correctness against NetworkX
    is_nx_correct = verify_pagerank_correctness(compare_results, tolerance=1e-4)
    print(f"NetworkX comparison correctness check: {'PASSED' if is_nx_correct else 'FAILED'}")
    
    # Print timing summary if requested
    if args.timing and args.verbose:
        print("\nTiming Summary:")
        print("=" * 60)
        print(f"{'Implementation':<30} {'Average Time (s)':<20}")
        print("-" * 60)
        for name, time_taken in time_results.items():
            print(f"{name:<30} {time_taken:.6f}")

if __name__ == '__main__':
    main()