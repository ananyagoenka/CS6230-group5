#!/usr/bin/env python3
"""
Script to run Connected Components benchmarks comparing traditional, 
multiprocessing, and linear algebra implementations
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import multiprocessing

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
    parser = argparse.ArgumentParser(description='Run Connected Components benchmarks')
    parser.add_argument('--sizes', type=int, nargs='+', default=[5000, 10000, 15000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--processes', type=int, nargs='+', default=[2, 4, 8], 
                        help='Number of processes for multiprocessing implementations')
    parser.add_argument('--threads', type=int, nargs='+', default=[2, 4, 8], 
                        help='Number of threads for OpenMP implementations')
    parser.add_argument('--skip-mp', action='store_true', help='Skip multiprocessing benchmarks')
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
    print("Using device: {}".format(device))
    
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
    print("System has {} CPU cores available".format(max_cores))
    
    # Filter process counts based on available cores
    processes_to_test = [p for p in args.processes if p <= max_cores]
    threads_to_test = [t for t in args.threads if t <= max_cores]
    print("Will test with process counts: {}".format(processes_to_test))
    print("Will test with thread counts: {}".format(threads_to_test))
    
    # Store results for summary
    all_results = {}
    
    # Verify implementation correctness if requested
    if args.verify:
        print("\nVerifying implementation correctness with graph of size {}...".format(args.verify_size))
        
        # Generate verification graph
        G_verify = generate_graph(args.verify_size)
        print_graph_stats(G_verify)
        
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
        
        # Run multiprocessing CC with a reasonable process count
        if not args.skip_mp:
            process_count = min(4, max_cores)
            print("Running multiprocessing CC with {} processes for verification...".format(process_count))
            try:
                _, mp_result = run_test(
                    ConnectedComponents.traditional_cc_multiprocessing,
                    adj_list_verify,
                    process_count,
                    n_runs=1
                )
                verification_results["MP_CC_{}processes".format(process_count)] = mp_result
            except Exception as e:
                print("Error in multiprocessing CC verification: {}".format(e))
        
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
            print("Error in LA CC verification: {}".format(e))
        
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
                        print("Mismatch in {} for nodes {} and {}".format(name, node1, node2))
                        component_match = False
                        all_correct = False
                        break
                
                if not component_match:
                    break
            
            # Check number of components
            if len(reference_components) != len(components):
                print("Number of components mismatch in {}: Expected {}, Got {}".format(
                    name, len(reference_components), len(components)))
                all_correct = False
            
            if component_match and len(reference_components) == len(components):
                print("{} matches the reference implementation".format(name))
        
        if not all_correct:
            print("\nWARNING: Some implementations do not match the reference!")
            proceed = input("Do you want to proceed with benchmarking anyway? (y/n): ")
            if proceed.lower() != 'y':
                print("Exiting...")
                return
    
    # Run benchmarks for each graph size
    for size in args.sizes:
        print("\n{}".format('-'*40))
        print("Benchmarking graph of size {}...".format(size))
        print("{}".format('-'*40))
        
        # Generate graph
        G = generate_graph(size)
        print_graph_stats(G)
        
        # Convert graph to different representations
        adj_list = graph_to_adj_list(G)
        adj_matrix_np = graph_to_adj_matrix_numpy(G)
        
        if args.gpu:
            adj_matrix_torch = graph_to_adj_matrix_torch(G, device=device)
            adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
        
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
        print("Average time: {:.6f} seconds".format(trad_result['avg_time']))
        print("Std dev: {:.6f} seconds".format(trad_result['std_time']))
        print("Min time: {:.6f} seconds".format(trad_result['min_time']))
        print("Max time: {:.6f} seconds".format(trad_result['max_time']))
        
        # Run OpenMP-based CC with different thread counts
        print("\nRunning OpenMP CC implementation...")
        for thread_count in threads_to_test:
            print("  With {} threads:".format(thread_count))
            try:
                openmp_result, _ = run_test(
                    ConnectedComponents.traditional_cc_openmp,
                    adj_list,
                    thread_count,
                    n_runs=args.runs
                )
                size_results["OpenMP_CC_{}threads".format(thread_count)] = openmp_result
                print("  Average time: {:.6f} seconds".format(openmp_result['avg_time']))
                print("  Std dev: {:.6f} seconds".format(openmp_result['std_time']))
                print("  Min time: {:.6f} seconds".format(openmp_result['min_time']))
                print("  Max time: {:.6f} seconds".format(openmp_result['max_time']))
                print("  Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / openmp_result['avg_time']))
            except Exception as e:
                print("  Error in OpenMP CC with {} threads: {}".format(thread_count, e))
        
        # Run multiprocessing-based CC with different process counts
        if not args.skip_mp:
            print("\nRunning multiprocessing CC implementation...")
            
            # Test with different process counts
            for process_count in processes_to_test:
                print("  With {} processes:".format(process_count))
                
                # Multiprocessing CC
                try:
                    mp_result, _ = run_test(
                        ConnectedComponents.traditional_cc_multiprocessing,
                        adj_list,
                        process_count,
                        n_runs=args.runs
                    )
                    size_results["MP_CC_{}processes".format(process_count)] = mp_result
                    print("  Average time: {:.6f} seconds".format(mp_result['avg_time']))
                    print("  Std dev: {:.6f} seconds".format(mp_result['std_time']))
                    print("  Min time: {:.6f} seconds".format(mp_result['min_time']))
                    print("  Max time: {:.6f} seconds".format(mp_result['max_time']))
                    print("  Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / mp_result['avg_time']))
                except Exception as e:
                    print("  Error in multiprocessing CC with {} processes: {}".format(process_count, e))
        
        # Run Numba-based CC
        print("\nRunning Numba CC implementation...")
        try:
            numba_result, _ = run_test(
                ConnectedComponents.traditional_cc_numba,
                adj_list,
                n_runs=args.runs
            )
            size_results['Numba_CC'] = numba_result
            print("Average time: {:.6f} seconds".format(numba_result['avg_time']))
            print("Std dev: {:.6f} seconds".format(numba_result['std_time']))
            print("Min time: {:.6f} seconds".format(numba_result['min_time']))
            print("Max time: {:.6f} seconds".format(numba_result['max_time']))
            print("Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / numba_result['avg_time']))
        except Exception as e:
            print("Error in Numba CC implementation: {}".format(e))
        
        # Run linear algebra CC on CPU
        print("\nRunning linear algebra CC on CPU...")
        la_cpu_result, _ = run_test(
            ConnectedComponents.la_cc_cpu,
            adj_matrix_np,
            n_runs=args.runs
        )
        size_results['LA_CC_CPU'] = la_cpu_result
        print("Average time: {:.6f} seconds".format(la_cpu_result['avg_time']))
        print("Std dev: {:.6f} seconds".format(la_cpu_result['std_time']))
        print("Min time: {:.6f} seconds".format(la_cpu_result['min_time']))
        print("Max time: {:.6f} seconds".format(la_cpu_result['max_time']))
        print("Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / la_cpu_result['avg_time']))
        
        # Run linear algebra CC on GPU if requested
        if args.gpu:
            print("\nRunning linear algebra CC on GPU (dense)...")
            la_gpu_result, _ = run_test(
                ConnectedComponents.la_cc_gpu,
                adj_matrix_torch,
                n_runs=args.runs
            )
            size_results['LA_CC_GPU_Dense'] = la_gpu_result
            print("Average time: {:.6f} seconds".format(la_gpu_result['avg_time']))
            print("Std dev: {:.6f} seconds".format(la_gpu_result['std_time']))
            print("Min time: {:.6f} seconds".format(la_gpu_result['min_time']))
            print("Max time: {:.6f} seconds".format(la_gpu_result['max_time']))
            print("Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / la_gpu_result['avg_time']))
            print("Speedup vs LA CPU: {:.2f}x".format(la_cpu_result['avg_time'] / la_gpu_result['avg_time']))
            
            print("\nRunning linear algebra CC on GPU (sparse)...")
            la_sparse_result, _ = run_test(
                ConnectedComponents.la_cc_sparse_gpu,
                adj_matrix_sparse,
                n_runs=args.runs
            )
            size_results['LA_CC_GPU_Sparse'] = la_sparse_result
            print("Average time: {:.6f} seconds".format(la_sparse_result['avg_time']))
            print("Std dev: {:.6f} seconds".format(la_sparse_result['std_time']))
            print("Min time: {:.6f} seconds".format(la_sparse_result['min_time']))
            print("Max time: {:.6f} seconds".format(la_sparse_result['max_time']))
            print("Speedup vs traditional: {:.2f}x".format(trad_result['avg_time'] / la_sparse_result['avg_time']))
            print("Speedup vs LA CPU: {:.2f}x".format(la_cpu_result['avg_time'] / la_sparse_result['avg_time']))
            print("Speedup vs LA GPU Dense: {:.2f}x".format(la_gpu_result['avg_time'] / la_sparse_result['avg_time']))
        
        # Store results for this size
        all_results[size] = size_results
    
    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY FOR {} GRAPHS".format(args.graph_type.upper()))
    print("="*60)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)", "Speedup"]
    print("{:<10} {:<35} {:<15} {:<10} {:<15} {:<15} {:<10}".format(*headers))
    print("-" * 110)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        trad_time = size_results['Traditional_CC']['avg_time']
        
        # Print traditional first
        alg_name = 'Traditional_CC'
        result = size_results[alg_name]
        print("{:<10} {:<35} {:<15.6f} {:<10.6f} {:<15.6f} {:<15.6f} {:<10.2f}".format(
            size, alg_name, result['avg_time'], result['std_time'], 
            result['min_time'], result['max_time'], 1.0
        ))
        
        # Then print other algorithms
        for alg_name, result in size_results.items():
            if alg_name == 'Traditional_CC':
                continue
            speedup = trad_time / result['avg_time']
            print("{:<10} {:<35} {:<15.6f} {:<10.6f} {:<15.6f} {:<15.6f} {:<10.2f}".format(
                '', alg_name, result['avg_time'], result['std_time'], 
                result['min_time'], result['max_time'], speedup
            ))
        
        print("-" * 110)
        
    print("\nBenchmarking complete.")

if __name__ == '__main__':
    main()