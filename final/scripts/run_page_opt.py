#!/usr/bin/env python3
"""
Script to run PageRank benchmarks comparing traditional CPU, sparse GPU, 
and CUDA-optimized sparse GPU implementations
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import gc

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
        # Clear CUDA cache before each run for fair comparison
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        start_time = time.time()
        results = func(*args, **kwargs)
        
        # Make sure all CUDA operations are completed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        
        print(f"  Run {i+1}/{n_runs} completed in {run_time:.6f} seconds")
        
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
    parser = argparse.ArgumentParser(description='Run PageRank benchmarks with CUDA-optimized implementation')
    parser.add_argument('--sizes', type=int, nargs='+', 
                        default=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000], 
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
    parser.add_argument('--skip-traditional', action='store_true', help='Skip traditional CPU implementation')
    parser.add_argument('--max-traditional-size', type=int, default=100000, 
                        help='Maximum graph size for traditional implementation (skipped if larger)')
    
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
    
    # Define graph generation function based on type
    if args.graph_type == 'random':
        def generate_graph(n):
            # For really large graphs, adjust p to maintain reasonable edge count
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
        
        # Run optimized sparse GPU PageRank v2
        print("Running CUDA-optimized sparse PageRank on GPU for verification...")
        try:
            _, la_opt_sparse_result = run_test(
                PageRank.la_pagerank_sparse_gpu_optimized_v2_turbo,
                adj_matrix_sparse_verify,
                damping=args.damping,
                max_iterations=args.max_iters,
                tol=args.tolerance,
                n_runs=1
            )
            verification_results['LA_PageRank_GPU_Sparse_Optimized'] = la_opt_sparse_result
        except Exception as e:
            print(f"Error in CUDA-optimized sparse PageRank verification: {e}")
        
        # Verify results
        print("\nVerifying implementation correctness...")
        all_correct = True
        reference_ranks, _ = verification_results['Traditional_PageRank']
        
        for name, (ranks, _) in verification_results.items():
            if name == 'Traditional_PageRank':
                continue
            
            # Helper function to convert tensor ranks to dict if needed
            def ensure_ranks_dict(ranks, n):
                if isinstance(ranks, torch.Tensor):
                    return {i: float(ranks[i]) for i in range(n)}
                return ranks
                
            # Convert tensor ranks to dict if needed
            if isinstance(ranks, torch.Tensor):
                ranks_dict = ensure_ranks_dict(ranks, adj_matrix_sparse_verify.shape[0])
            else:
                ranks_dict = ranks
            
            # Check if all nodes in the reference have similar rank
            ranks_match = True
            max_diff = 0.0
            
            for node, rank in reference_ranks.items():
                if node not in ranks_dict:
                    print(f"Node {node} missing in {name}")
                    ranks_match = False
                    all_correct = False
                    break
                
                # Check if the ranks are similar within tolerance
                diff = abs(float(rank) - float(ranks_dict[node]))
                max_diff = max(max_diff, diff)
                
                if diff > 1e-4:  # Using a wider tolerance for PageRank comparison
                    print(f"Rank mismatch in {name} for node {node}: "
                          f"Expected {rank}, Got {ranks_dict[node]}, Diff {diff}")
                    ranks_match = False
                    all_correct = False
                    if max_diff > 1e-2:  # Only break on larger differences
                        break
            
            # Check if all nodes in this implementation are in the reference
            for node in ranks_dict:
                if node not in reference_ranks:
                    print(f"Node {node} in {name} but not in reference implementation")
                    ranks_match = False
                    all_correct = False
                    break
            
            if ranks_match:
                print(f"{name} matches the reference implementation (max diff: {max_diff:.8f})")
            else:
                print(f"{name} has differences from reference (max diff: {max_diff:.8f})")
        
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
            
            # Store results for this size
            size_results = {}
            
            # Skip traditional PageRank for very large graphs or if requested
            run_traditional = (
                not args.skip_traditional  
                # size <= args.max_traditional_size
            )
            
            if run_traditional:
                print("\nRunning traditional PageRank...")
                # Convert graph to adjacency list for traditional PageRank
                print("Converting graph to adjacency list...")
                adj_list = graph_to_adj_list(G)
                
                print("Running traditional PageRank algorithm...")
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
                        print(f"Average iterations: {trad_result['iterations']:.1f}")
                    
                    baseline_time = trad_result['avg_time']
                except Exception as e:
                    print(f"Error in traditional PageRank: {e}")
                    baseline_time = None
                
                # Free memory
                del adj_list
                gc.collect()
            else:
                if size > args.max_traditional_size:
                    print(f"\nSkipping traditional PageRank for large graph (size > {args.max_traditional_size})")
                else:
                    print("\nSkipping traditional PageRank as requested")
                baseline_time = None
            
            # Convert graph to sparse matrix for GPU methods (do this only once)
            print("\nConverting graph to sparse matrix for GPU implementations...")
            adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device=device)
            print("Sparse matrix conversion complete")
            
            # Free the original graph to save memory
            del G
            gc.collect()
            torch.cuda.empty_cache()
            
            # Run standard sparse GPU PageRank
            print("\nRunning standard sparse GPU PageRank...")
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
                    print(f"Average iterations: {la_sparse_result['iterations']:.1f}")
                
                if baseline_time:
                    print(f"Speedup vs traditional: {baseline_time / la_sparse_result['avg_time']:.2f}x")
                    sparse_time = la_sparse_result['avg_time']
                else:
                    sparse_time = la_sparse_result['avg_time']
                    # This becomes our baseline if traditional PageRank was skipped
                    baseline_time = sparse_time
            except Exception as e:
                print(f"Error in standard sparse GPU PageRank: {e}")
                sparse_time = None
            
            # Run optimized sparse GPU PageRank v2
            print("\nRunning CUDA-optimized sparse PageRank on GPU...")
            try:
                la_opt_sparse_result, _ = run_test(
                    PageRank.la_pagerank_sparse_gpu_optimized_v2_turbo,
                    adj_matrix_sparse,
                    damping=args.damping,
                    max_iterations=args.max_iters,
                    tol=args.tolerance,
                    n_runs=args.runs
                )
                size_results['LA_PageRank_GPU_Sparse_Optimized'] = la_opt_sparse_result
                print(f"Average time: {la_opt_sparse_result['avg_time']:.6f} seconds")
                print(f"Std dev: {la_opt_sparse_result['std_time']:.6f} seconds")
                print(f"Min time: {la_opt_sparse_result['min_time']:.6f} seconds")
                print(f"Max time: {la_opt_sparse_result['max_time']:.6f} seconds")
                if 'iterations' in la_opt_sparse_result:
                    print(f"Average iterations: {la_opt_sparse_result['iterations']:.1f}")
                
                if baseline_time:
                    print(f"Speedup vs traditional: {baseline_time / la_opt_sparse_result['avg_time']:.2f}x")
                if sparse_time:
                    print(f"Speedup vs standard sparse: {sparse_time / la_opt_sparse_result['avg_time']:.2f}x")
            except Exception as e:
                print(f"Error in CUDA-optimized sparse PageRank: {e}")
            
            # Store results for this size
            all_results[size] = size_results
            
            # Clean up to free memory for next iteration
            del adj_matrix_sparse
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing graph of size {size}: {e}")
            print("Skipping to next size...")
            continue
    
    # Print summary of all results
    print("\n" + "="*80)
    print(f"SUMMARY FOR {args.graph_type.upper()} GRAPHS")
    print("="*80)
    
    # Print header
    headers = ["Size", "Algorithm", "Avg Time (s)", "Std Dev", "Iterations", "Min Time (s)", "Max Time (s)", "Speedup vs Trad", "Speedup vs Sparse"]
    print(f"{headers[0]:<10} {headers[1]:<35} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15} {headers[8]:<15}")
    print("-" * 140)
    
    # Print results for each size
    for size in sorted(all_results.keys()):
        size_results = all_results[size]
        
        # Get baseline times if available
        trad_time = size_results.get('Traditional_PageRank', {}).get('avg_time', None)
        sparse_time = size_results.get('LA_PageRank_GPU_Sparse', {}).get('avg_time', None)
        
        # If traditional not available, use sparse as baseline
        if trad_time is None and sparse_time is not None:
            trad_time = sparse_time
        
        # Print results for each algorithm
        algorithms = [
            'Traditional_PageRank', 
            'LA_PageRank_GPU_Sparse', 
            'LA_PageRank_GPU_Sparse_Optimized'
        ]
        
        for i, alg_name in enumerate(algorithms):
            if alg_name not in size_results:
                continue
                
            result = size_results[alg_name]
            
            # Calculate speedups
            if trad_time and alg_name != 'Traditional_PageRank':
                speedup_vs_trad = trad_time / result['avg_time']
            else:
                speedup_vs_trad = 1.0 if alg_name == 'Traditional_PageRank' else float('nan')
            
            if sparse_time and alg_name != 'LA_PageRank_GPU_Sparse':
                speedup_vs_sparse = sparse_time / result['avg_time']
            else:
                speedup_vs_sparse = 1.0 if alg_name == 'LA_PageRank_GPU_Sparse' else float('nan')
            
            # Print size only for the first algorithm
            size_str = str(size) if i == 0 else ''
            
            # Get iterations if available
            iter_str = f"{result.get('iterations', 0):.1f}" if 'iterations' in result else "N/A"
            
            # Format speedup values, handling NaN
            speedup_trad_str = f"{speedup_vs_trad:.2f}x" if not np.isnan(speedup_vs_trad) else "N/A"
            speedup_sparse_str = f"{speedup_vs_sparse:.2f}x" if not np.isnan(speedup_vs_sparse) else "N/A"
            
            print(f"{size_str:<10} {alg_name:<35} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
                  f"{iter_str:<10} {result['min_time']:<15.6f} {result['max_time']:<15.6f} "
                  f"{speedup_trad_str:<15} {speedup_sparse_str:<15}")
        
        print("-" * 140)
    
    print("\nBenchmark complete!")

if __name__ == '__main__':
    main()