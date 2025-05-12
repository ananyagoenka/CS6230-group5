#!/usr/bin/env python3
"""
Script to verify correctness of Connected Components implementations
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

from src.algorithms.connected_components import ConnectedComponents
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

def verify_cc_correctness(cc_results):
    """
    Verify that all Connected Components implementations produce the same results
    
    Parameters:
    -----------
    cc_results : dict
        Dictionary mapping implementation names to CC results (components, component_map)
        
    Returns:
    --------
    is_correct : bool
        True if all implementations produce identical results
    """
    if not cc_results:
        return True
    
    # Get reference implementation (first in the dictionary)
    reference_name = list(cc_results.keys())[0]
    reference_components, reference_component_map = cc_results[reference_name]
    
    all_correct = True
    
    for name, (components, component_map) in cc_results.items():
        if name == reference_name:
            continue
        
        # Check if the component maps are equivalent (same connectivity)
        # Two nodes should be in the same component in both implementations
        component_match = True
        for node1 in reference_component_map:
            for node2 in reference_component_map:
                # Check if nodes are in the same component in both implementations
                ref_same_component = (reference_component_map[node1] == reference_component_map[node2])
                test_same_component = (component_map[node1] == component_map[node2])
                
                if ref_same_component != test_same_component:
                    print("Mismatch in {} for nodes {} and {}: Reference: {}, Test: {}".format(name, node1, node2, ref_same_component, test_same_component))
                    component_match = False
                    all_correct = False
                    break
            
            if not component_match:
                break
        
        # Check number of components
        ref_num_components = len(reference_components)
        test_num_components = len(components)
        if ref_num_components != test_num_components:
            print("Number of components mismatch in {}: Reference: {}, Test: {}".format(name, ref_num_components, test_num_components))
            all_correct = False
        
        # If all checks passed, the implementation matches the reference
        if component_match and ref_num_components == test_num_components:
            print("{} matches the reference implementation".format(name))
    
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
    
    # CC specific statistics
    component_counts = {}
    largest_component_sizes = {}
    for name, (components, _) in results.items():
        component_counts[name] = len(components)
        largest_component_sizes[name] = max(len(component) for component in components) if components else 0
    
    # Print header with timing info if available
    if time_results:
        print("{:<35} {:<20} {:<20} {:<15}".format("Implementation", "Number of Components", "Largest Component Size", "Time (s)"))
    else:
        print("{:<35} {:<20} {:<20}".format("Implementation", "Number of Components", "Largest Component Size"))
    print("-" * 80)
    
    # Print stats for each implementation
    for name in results:
        if time_results:
            time_taken = time_results.get(name, "N/A")
            if isinstance(time_taken, (int, float)):
                time_str = "{:.6f}".format(time_taken)
            else:
                time_str = time_taken
            print("{:<35} {:<20} {:<20} {:<15}".format(name, component_counts[name], largest_component_sizes[name], time_str))
        else:
            print("{:<35} {:<20} {:<20}".format(name, component_counts[name], largest_component_sizes[name]))
    
    # Calculate speedup relative to sequential if timing info is available
    if time_results and "Traditional_CC" in time_results:
        baseline_time = time_results["Traditional_CC"]
        if isinstance(baseline_time, (int, float)) and baseline_time > 0:
            print("\nSpeedup relative to sequential implementation:")
            for name, time_taken in time_results.items():
                if name != "Traditional_CC" and isinstance(time_taken, (int, float)):
                    speedup = baseline_time / time_taken
                    print("{:<35} {:.2f}x".format(name, speedup))

def main():
    parser = argparse.ArgumentParser(description='Verify correctness of Connected Components implementations')
    parser.add_argument('--size', type=int, default=1000, 
                        help='Graph size for verification')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--verbose', action='store_true', help='Print detailed statistics')
    parser.add_argument('--processes', type=int, nargs='+', default=[2, 4], 
                        help='Process counts to test for multiprocessing implementations')
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
    print("Using device: {}".format(device))
    
    # Get available cores
    available_cores = multiprocessing.cpu_count()
    print("System has {} CPU cores available".format(available_cores))
    
    # Use only process counts that are available
    processes_to_test = [p for p in args.processes if p <= available_cores]
    print("Testing with process counts: {}".format(processes_to_test))
    
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
    
    print("\nGenerating {} graph of size {}...".format(args.graph_type, args.size))
    
    # Generate graph
    G = generate_graph(args.size)
    print_graph_stats(G)
    
    # Convert graph to different representations
    adj_list = graph_to_adj_list(G)
    adj_matrix_np = graph_to_adj_matrix_numpy(G)
    
    if args.gpu:
        adj_matrix_torch = graph_to_adj_matrix_torch(G, device=device)
        adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
    
    # Run all Connected Components implementations
    cc_results = {}
    time_results = {} if args.timing else None
    
    print("\nRunning Connected Components implementations...")
    
    # Traditional CC (baseline)
    print("Running traditional CC...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            traditional_result = ConnectedComponents.traditional_cc_cpu(adj_list)
        time_results["Traditional_CC"] = (time.time() - start_time) / args.repeat
    else:
        traditional_result = ConnectedComponents.traditional_cc_cpu(adj_list)
    cc_results['Traditional_CC'] = traditional_result
    
    # OpenMP CC
    print("Running OpenMP CC...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            openmp_result = ConnectedComponents.traditional_cc_openmp(adj_list, num_threads=4)
        time_results["OpenMP_CC"] = (time.time() - start_time) / args.repeat
    else:
        openmp_result = ConnectedComponents.traditional_cc_openmp(adj_list, num_threads=4)
    cc_results['OpenMP_CC'] = openmp_result
    
    # Multiprocessing CC with different process counts
    print("Running Multiprocessing CC implementations...")
    for process_count in processes_to_test:
        print("  With {} processes:".format(process_count))
        if args.timing:
            start_time = time.time()
            for _ in range(args.repeat):
                mp_result = ConnectedComponents.traditional_cc_multiprocessing(adj_list, process_count)
            time_results["Multiprocessing_CC_{}processes".format(process_count)] = (time.time() - start_time) / args.repeat
        else:
            mp_result = ConnectedComponents.traditional_cc_multiprocessing(adj_list, process_count)
        cc_results["Multiprocessing_CC_{}processes".format(process_count)] = mp_result
    
    # Numba CC
    print("Running Numba CC...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            numba_result = ConnectedComponents.traditional_cc_numba(adj_list)
        time_results["Numba_CC"] = (time.time() - start_time) / args.repeat
    else:
        numba_result = ConnectedComponents.traditional_cc_numba(adj_list)
    cc_results['Numba_CC'] = numba_result
    
    # Linear algebra CC on CPU
    print("Running linear algebra CC on CPU...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            la_cpu_result = ConnectedComponents.la_cc_cpu(adj_matrix_np)
        time_results["LA_CC_CPU"] = (time.time() - start_time) / args.repeat
    else:
        la_cpu_result = ConnectedComponents.la_cc_cpu(adj_matrix_np)
    cc_results['LA_CC_CPU'] = la_cpu_result
    
    # Linear algebra CC on GPU if requested
    if args.gpu:
        print("Running linear algebra CC on GPU (dense)...")
        if args.timing:
            start_time = time.time()
            for _ in range(args.repeat):
                la_gpu_result = ConnectedComponents.la_cc_gpu(adj_matrix_torch)
            time_results["LA_CC_GPU_Dense"] = (time.time() - start_time) / args.repeat
        else:
            la_gpu_result = ConnectedComponents.la_cc_gpu(adj_matrix_torch)
        cc_results['LA_CC_GPU_Dense'] = la_gpu_result
        
        print("Running linear algebra CC on GPU (sparse)...")
        if args.timing:
            start_time = time.time()
            for _ in range(args.repeat):
                la_sparse_result = ConnectedComponents.la_cc_sparse_gpu(adj_matrix_sparse)
            time_results["LA_CC_GPU_Sparse"] = (time.time() - start_time) / args.repeat
        else:
            la_sparse_result = ConnectedComponents.la_cc_sparse_gpu(adj_matrix_sparse)
        cc_results['LA_CC_GPU_Sparse'] = la_sparse_result
    
    # Compare with NetworkX implementation
    print("\nComparing with NetworkX implementation...")
    if args.timing:
        start_time = time.time()
        for _ in range(args.repeat):
            nx_components = list(nx.connected_components(G))
            # Create the component map
            nx_component_map = {}
            for i, component in enumerate(nx_components):
                for node in component:
                    nx_component_map[node] = i
        time_results['NetworkX'] = (time.time() - start_time) / args.repeat
    else:
        nx_components = list(nx.connected_components(G))
        # Create the component map
        nx_component_map = {}
        for i, component in enumerate(nx_components):
            for node in component:
                nx_component_map[node] = i
    
    # Convert sets to lists for consistent format
    nx_components_list = [list(component) for component in nx_components]
    nx_result = (nx_components_list, nx_component_map)
    cc_results['NetworkX'] = nx_result
    
    # Verify correctness
    print("\nVerifying correctness...")
    is_correct = verify_cc_correctness(cc_results)
    print("Connected Components implementations correctness check: {}".format('PASSED' if is_correct else 'FAILED'))
    
    # Print detailed statistics if requested
    if args.verbose or args.timing:
        print_algorithm_stats(cc_results, time_results)

if __name__ == '__main__':
    main()