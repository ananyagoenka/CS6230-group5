#!/usr/bin/env python3
"""
Script to run BFS benchmarks comparing traditional CPU, sparse GPU, 
and CUDA-optimized sparse GPU implementations with no size limitations
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import gc
import psutil

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
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

def print_memory_usage():
    """Print current CPU and GPU memory usage"""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"CPU Memory Usage: {cpu_mem:.2f} MB")
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        print(f"GPU Memory Allocated: {gpu_mem_allocated:.2f} MB")
        print(f"GPU Memory Reserved: {gpu_mem_reserved:.2f} MB")

def run_test(func, *args, n_runs=1, **kwargs):
    """Run a test function multiple times and return timing statistics"""
    times = []
    results = None
    
    for i in range(n_runs):
        # Clear CUDA cache before each run to ensure fair comparison
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection to free memory
        gc.collect()
            
        start_time = time.time()
        results = func(*args, **kwargs)
        # Ensure all CUDA operations are completed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        
        print(f"  Run {i+1}/{n_runs} completed in {run_time:.6f} seconds")
    
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
    parser = argparse.ArgumentParser(description='Run BFS benchmarks with CUDA-optimized implementation')
    parser.add_argument('--sizes', type=int, nargs='+', 
                        default=[500, 1000, 2500, 5000, 7500, 10000], 
                        help='Graph sizes to benchmark (no size limit)')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each benchmark')
    parser.add_argument('--verify', action='store_true', help='Verify implementation correctness before benchmarking')
    parser.add_argument('--verify-size', type=int, default=500, help='Graph size for verification')
    parser.add_argument('--skip-traditional', action='store_true', help='Skip traditional CPU implementation')
    parser.add_argument('--max-traditional-size', type=int, default=100000, 
                        help='Maximum graph size for traditional implementation (skipped if larger)')
    parser.add_argument('--only-gpu', action='store_true', help='Only run GPU implementations')
    parser.add_argument('--only-optimized', action='store_true', help='Only run optimized GPU implementation')
    
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
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # For large graphs, adjust PyTorch settings
    # Increase the allocation batch size
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Define graph generation function based on type
    if args.graph_type == 'random':
        def generate_graph(n):
            # For really large graphs, adjust p to maintain reasonable edge count
            # For n > 1M, we reduce the average degree to avoid memory issues
            if n > 1000000:
                p = 5 / (n - 1)  # Reduced avg degree
            else:
                p = 10 / (n - 1)  # Standard avg degree
            return generate_random_graph(n, p, seed=args.seed)
    elif args.graph_type == 'scale-free':
        def generate_graph(n):
            # Adjust m for very large graphs
            if n > 1000000:
                m = 3  # Each new node adds 3 edges for very large graphs
            else:
                m = 5  # Standard setting
            return generate_scale_free_graph(n, m, seed=args.seed)
    elif args.graph_type == 'small-world':
        def generate_graph(n):
            # Adjust k for very large graphs
            if n > 1000000:
                k = 6  # Reduced nearest neighbors for very large graphs
            else:
                k = 10  # Standard setting
            return generate_small_world_graph(n, k, 0.1, seed=args.seed)
    
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
        
        # Choose a start node (node with highest degree)
        start_node_verify = max(G_verify.degree(), key=lambda x: x[1])[0]
        
        verification_results = {}
        
        # Run traditional BFS
        if not args.only_gpu and not args.only_optimized:
            print("Running traditional BFS for verification...")
            _, traditional_result = run_test(
                BFS.traditional_bfs_cpu,
                adj_list_verify,
                start_node_verify,
                n_runs=1
            )
            verification_results['Traditional_BFS'] = traditional_result
        
        # Run sparse GPU BFS
        if not args.only_optimized:
            print("Running linear algebra BFS on GPU (sparse) for verification...")
            try:
                _, la_sparse_result = run_test(
                    BFS.la_bfs_sparse_gpu,
                    adj_matrix_sparse_verify,
                    start_node_verify,
                    n_runs=1
                )
                verification_results['LA_BFS_GPU_Sparse'] = la_sparse_result
            except Exception as e:
                print(f"Error in LA GPU Sparse BFS verification: {e}")
        
        # Run optimized sparse GPU BFS
        print("Running CUDA-optimized sparse BFS on GPU for verification...")
        try:
            _, la_opt_sparse_result = run_test(
                BFS.la_bfs_sparse_gpu_optimized_v2,
                adj_matrix_sparse_verify,
                start_node_verify,
                n_runs=1
            )
            verification_results['LA_BFS_GPU_Sparse_Optimized'] = la_opt_sparse_result
        except Exception as e:
            print(f"Error in CUDA-optimized sparse BFS verification: {e}")
        
        # Verify results
        print("\nVerifying implementation correctness...")
        all_correct = True
        reference_algo = None
        
        # Find reference algorithm (in order of preference)
        if 'Traditional_BFS' in verification_results:
            reference_algo = 'Traditional_BFS'
        elif 'LA_BFS_GPU_Sparse' in verification_results:
            reference_algo = 'LA_BFS_GPU_Sparse'
        elif 'LA_BFS_GPU_Sparse_Optimized' in verification_results:
            reference_algo = 'LA_BFS_GPU_Sparse_Optimized'
        
        if reference_algo is None:
            print("Error: No implementations completed verification successfully")
            return
            
        reference_visited, reference_distances = verification_results[reference_algo]
        print(f"Using {reference_algo} as reference implementation")
        
        for name, (visited, distances) in verification_results.items():
            if name == reference_algo:
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
        print(f"\n{'-'*60}")
        print(f"Benchmarking graph of size {size}...")
        print(f"{'-'*60}")
        
        try:
            # Generate graph
            print(f"Generating {args.graph_type} graph with {size} nodes...")
            G = generate_graph(size)
            print_simplified_graph_stats(G)
            
            # Choose a start node (node with highest degree)
            print("Finding start node with highest degree...")
            start_node = max(G.degree(), key=lambda x: x[1])[0]
            print(f"Start node: {start_node} with degree {G.degree(start_node)}")
            
            # Print current memory usage
            print("\nMemory usage before running algorithms:")
            print_memory_usage()
            
            # Store results for this size
            size_results = {}
            
            # Skip traditional BFS for very large graphs or if requested
            run_traditional = (
                not args.skip_traditional and 
                not args.only_gpu and 
                not args.only_optimized and 
                size <= args.max_traditional_size
            )
            
            if run_traditional:
                print("\nRunning traditional BFS...")
                # Convert graph to adjacency list for traditional BFS
                print("Converting graph to adjacency list...")
                adj_list = graph_to_adj_list(G)
                
                print("Running traditional BFS algorithm...")
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
                baseline_time = trad_result['avg_time']
                
                # Free memory
                del adj_list
                gc.collect()
            else:
                if size > args.max_traditional_size:
                    print(f"\nSkipping traditional BFS for large graph (size > {args.max_traditional_size})")
                else:
                    print("\nSkipping traditional BFS as requested")
                baseline_time = None
            
            # Convert graph to sparse matrix for GPU methods (do this only once)
            print("\nConverting graph to sparse matrix for GPU implementations...")
            adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
            print("Sparse matrix conversion complete")
            
            # Print memory usage after conversion
            print("\nMemory usage after sparse matrix conversion:")
            print_memory_usage()
            
            # Free the original graph to save memory
            del G
            gc.collect()
            torch.cuda.empty_cache()
            
            # Run standard sparse GPU BFS if not skipped
            if not args.only_optimized:
                print("\nRunning linear algebra BFS on GPU (sparse)...")
                la_sparse_result, _ = run_test(
                    BFS.la_bfs_sparse_gpu,
                    adj_matrix_sparse,
                    start_node,
                    n_runs=args.runs
                )
                size_results['LA_BFS_GPU_Sparse'] = la_sparse_result
                print(f"Average time: {la_sparse_result['avg_time']:.6f} seconds")
                print(f"Std dev: {la_sparse_result['std_time']:.6f} seconds")
                print(f"Min time: {la_sparse_result['min_time']:.6f} seconds")
                print(f"Max time: {la_sparse_result['max_time']:.6f} seconds")
                
                if baseline_time:
                    print(f"Speedup vs traditional: {baseline_time / la_sparse_result['avg_time']:.2f}x")
                    sparse_time = la_sparse_result['avg_time']
                else:
                    sparse_time = la_sparse_result['avg_time']
                    # This becomes our baseline if traditional BFS was skipped
                    baseline_time = sparse_time
            else:
                print("\nSkipping standard sparse GPU BFS as requested")
                sparse_time = None
            
            # Run CUDA-optimized sparse GPU BFS
            print("\nRunning CUDA-optimized sparse BFS on GPU...")
            la_opt_sparse_result, _ = run_test(
                BFS.la_bfs_sparse_gpu_optimized_v2,
                adj_matrix_sparse,
                start_node,
                n_runs=args.runs
            )
            size_results['LA_BFS_GPU_Sparse_Optimized'] = la_opt_sparse_result
            print(f"Average time: {la_opt_sparse_result['avg_time']:.6f} seconds")
            print(f"Std dev: {la_opt_sparse_result['std_time']:.6f} seconds")
            print(f"Min time: {la_opt_sparse_result['min_time']:.6f} seconds")
            print(f"Max time: {la_opt_sparse_result['max_time']:.6f} seconds")
            
            if baseline_time:
                print(f"Speedup vs traditional: {baseline_time / la_opt_sparse_result['avg_time']:.2f}x")
            
            if sparse_time:
                print(f"Speedup vs standard sparse: {sparse_time / la_opt_sparse_result['avg_time']:.2f}x")
            
            # Store results for this size
            all_results[size] = size_results
            
            # Clean up to free memory for next iteration
            del adj_matrix_sparse
            gc.collect()
            torch.cuda.empty_cache()
            
            # Print memory usage after tests
            print("\nMemory usage after completing tests:")
            print_memory_usage()
            
        except Exception as e:
            print(f"\nError processing graph of size {size}: {e}")
            print("Skipping to next size...")
            continue
    
    # Print summary of all results
    print("\n" + "="*80)
    print(f"SUMMARY FOR {args.graph_type.upper()} GRAPHS")
    print("="*80)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)", "Speedup vs Trad", "Speedup vs Sparse"]
    print(f"{headers[0]:<10} {headers[1]:<30} {headers[2]:<15} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15}")
    print("-" * 125)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        
        # Get baseline times if available
        trad_time = size_results.get('Traditional_BFS', {}).get('avg_time', None)
        sparse_time = size_results.get('LA_BFS_GPU_Sparse', {}).get('avg_time', None)
        
        # If traditional not available, use sparse as baseline
        if trad_time is None and sparse_time is not None:
            trad_time = sparse_time
        
        # If sparse not available but optimized is, use optimized time for sparse comparisons
        if sparse_time is None and 'LA_BFS_GPU_Sparse_Optimized' in size_results:
            sparse_time = size_results['LA_BFS_GPU_Sparse_Optimized']['avg_time']
        
        # Print results for each algorithm
        algorithms = ['Traditional_BFS', 'LA_BFS_GPU_Sparse', 'LA_BFS_GPU_Sparse_Optimized']
        for i, alg_name in enumerate(algorithms):
            if alg_name not in size_results:
                continue
                
            result = size_results[alg_name]
            
            # Calculate speedups
            if trad_time and alg_name != 'Traditional_BFS':
                speedup_vs_trad = trad_time / result['avg_time']
            else:
                speedup_vs_trad = 1.0 if alg_name == 'Traditional_BFS' else float('nan')
            
            if sparse_time and alg_name != 'LA_BFS_GPU_Sparse':
                speedup_vs_sparse = sparse_time / result['avg_time']
            else:
                speedup_vs_sparse = 1.0 if alg_name == 'LA_BFS_GPU_Sparse' else float('nan')
            
            # Print size only for the first algorithm
            size_str = str(size) if i == 0 else ''
            
            # Format speedup values, handling NaN
            speedup_trad_str = f"{speedup_vs_trad:.2f}x" if not np.isnan(speedup_vs_trad) else "N/A"
            speedup_sparse_str = f"{speedup_vs_sparse:.2f}x" if not np.isnan(speedup_vs_sparse) else "N/A"
            
            print(f"{size_str:<10} {alg_name:<30} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
                  f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {speedup_trad_str:<15} {speedup_sparse_str:<15}")
        
        print("-" * 125)
    
    print("\nBenchmark complete!")

if __name__ == '__main__':
    main()