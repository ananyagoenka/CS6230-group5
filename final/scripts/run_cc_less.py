#!/usr/bin/env python3
"""
Script to run Connected Components benchmarks comparing traditional and 
linear algebra implementations (CPU and GPU) - without plotting or saving
"""

import os
import sys
import argparse
import torch
import numpy as np
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
    graph_to_adj_matrix_torch
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
    parser = argparse.ArgumentParser(description='Run Connected Components benchmarks (excluding multiprocessing)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[2000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--verify', action='store_true', help='Verify implementation correctness before benchmarking')
    parser.add_argument('--verify-size', type=int, default=500, help='Graph size for verification')
    
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
        adj_matrix_np_verify = graph_to_adj_matrix_numpy(G_verify)
        
        verification_results = {}
        
        # Run traditional CC
        print("Running traditional CC for verification...")
        _, traditional_result = run_test(
            ConnectedComponents.traditional_cc_cpu,
            adj_list_verify,
            n_runs=1
        )
        verification_results['Traditional_CC'] = traditional_result
        
        # Run linear algebra CC on CPU
        print("Running linear algebra CC on CPU for verification...")
        try:
            _, la_cpu_result = run_test(
                ConnectedComponents.la_cc_cpu,
                adj_matrix_np_verify,
                n_runs=1
            )
            verification_results['LA_CC_CPU'] = la_cpu_result
        except Exception as e:
            print(f"Error in LA CC verification: {e}")
        
        # Run linear algebra CC on GPU if requested
        if args.gpu:
            # Prepare GPU tensors
            adj_matrix_torch_verify = graph_to_adj_matrix_torch(G_verify, device=device)
            
            print("Running linear algebra CC on GPU (dense) for verification...")
            try:
                _, la_gpu_result = run_test(
                    ConnectedComponents.la_cc_gpu,
                    adj_matrix_torch_verify,
                    n_runs=1
                )
                verification_results['LA_CC_GPU_Dense'] = la_gpu_result
            except Exception as e:
                print(f"Error in LA GPU Dense CC verification: {e}")
        
        # Verify results
        print("\nVerifying implementation correctness...")
        all_correct = True
        reference_components, reference_component_map = verification_results['Traditional_CC']
        
        for name, (components, component_map) in verification_results.items():
            if name == 'Traditional_CC':
                continue
            
            # Check if the component maps are equivalent (same connectivity)
            component_match = True
            for node1 in reference_component_map:
                for node2 in reference_component_map:
                    # Check if nodes are in the same component in both implementations
                    ref_same_component = (reference_component_map[node1] == reference_component_map[node2])
                    test_same_component = (component_map[node1] == component_map[node2])
                    
                    if ref_same_component != test_same_component:
                        print(f"Mismatch in {name} for nodes {node1} and {node2}")
                        component_match = False
                        all_correct = False
                        break
                
                if not component_match:
                    break
            
            # Check number of components
            if len(reference_components) != len(components):
                print(f"Number of components mismatch in {name}: Expected {len(reference_components)}, Got {len(components)}")
                all_correct = False
            
            if component_match and len(reference_components) == len(components):
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
        adj_matrix_np = graph_to_adj_matrix_numpy(G)
        
        if args.gpu:
            adj_matrix_torch = graph_to_adj_matrix_torch(G, device=device)
        
        # Store results for this size
        size_results = {}
        
        # Run traditional CC
        print("\nRunning traditional CC...")
        trad_result, _ = run_test(
            ConnectedComponents.traditional_cc_cpu,
            adj_list,
            n_runs=args.runs
        )
        size_results['Traditional_CC'] = trad_result
        print(f"Average time: {trad_result['avg_time']:.6f} seconds")
        print(f"Std dev: {trad_result['std_time']:.6f} seconds")
        print(f"Min time: {trad_result['min_time']:.6f} seconds")
        print(f"Max time: {trad_result['max_time']:.6f} seconds")
        
        # Run linear algebra CC on CPU
        print("\nRunning linear algebra CC on CPU...")
        la_cpu_result, _ = run_test(
            ConnectedComponents.la_cc_cpu,
            adj_matrix_np,
            n_runs=args.runs
        )
        size_results['LA_CC_CPU'] = la_cpu_result
        print(f"Average time: {la_cpu_result['avg_time']:.6f} seconds")
        print(f"Std dev: {la_cpu_result['std_time']:.6f} seconds")
        print(f"Min time: {la_cpu_result['min_time']:.6f} seconds")
        print(f"Max time: {la_cpu_result['max_time']:.6f} seconds")
        print(f"Speedup vs traditional: {trad_result['avg_time'] / la_cpu_result['avg_time']:.2f}x")
        
        # Run linear algebra CC on GPU if requested
        if args.gpu:
            print("\nRunning linear algebra CC on GPU (dense)...")
            la_gpu_result, _ = run_test(
                ConnectedComponents.la_cc_gpu,
                adj_matrix_torch,
                n_runs=args.runs
            )
            size_results['LA_CC_GPU_Dense'] = la_gpu_result
            print(f"Average time: {la_gpu_result['avg_time']:.6f} seconds")
            print(f"Std dev: {la_gpu_result['std_time']:.6f} seconds")
            print(f"Min time: {la_gpu_result['min_time']:.6f} seconds")
            print(f"Max time: {la_gpu_result['max_time']:.6f} seconds")
            print(f"Speedup vs traditional: {trad_result['avg_time'] / la_gpu_result['avg_time']:.2f}x")
            print(f"Speedup vs LA CPU: {la_cpu_result['avg_time'] / la_gpu_result['avg_time']:.2f}x")
        
        # Store results for this size
        all_results[size] = size_results
    
    # Print summary of all results
    print("\n" + "="*60)
    print(f"SUMMARY FOR {args.graph_type.upper()} GRAPHS")
    print("="*60)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)", "Speedup"]
    print(f"{headers[0]:<10} {headers[1]:<35} {headers[2]:<15} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15} {headers[6]:<10}")
    print("-" * 110)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        trad_time = size_results['Traditional_CC']['avg_time']
        
        # Print traditional first
        alg_name = 'Traditional_CC'
        result = size_results[alg_name]
        print(f"{size:<10} {alg_name:<35} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
              f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {1.0:<10.2f}")
        
        # Then print other algorithms
        for alg_name, result in size_results.items():
            if alg_name == 'Traditional_CC':
                continue
            speedup = trad_time / result['avg_time']
            print(f"{'':<10} {alg_name:<35} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
                  f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {speedup:<10.2f}")
        
        print("-" * 110)
        
    print("\nBenchmarking complete.")

if __name__ == '__main__':
    main()