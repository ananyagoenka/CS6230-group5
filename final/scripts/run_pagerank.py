#!/usr/bin/env python3
"""
Script to run PageRank benchmarks comparing traditional, multiprocessing, 
and linear algebra implementations
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
import multiprocessing

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
from src.utils.benchmark import Benchmark

def main():
    parser = argparse.ArgumentParser(description='Run PageRank benchmarks')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs for each benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor for PageRank')
    parser.add_argument('--max-iters', type=int, default=100, help='Maximum iterations for PageRank')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Convergence tolerance for PageRank')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--processes', type=int, nargs='+', default=[2, 4, 8], 
                        help='Number of processes for multiprocessing implementations')
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
    print(f"Using device: {device}")
    
    # Create benchmark object
    benchmark = Benchmark(f'pagerank_{args.graph_type}', save_dir=args.save_dir)
    
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
    
    # Verify implementation correctness if requested
    if args.verify:
        print(f"\nVerifying implementation correctness with graph of size {args.verify_size}...")
        
        # Generate verification graph
        G_verify = generate_graph(args.verify_size)
        print_graph_stats(G_verify)
        
        # Convert graph to different representations
        adj_list_verify = graph_to_adj_list(G_verify)
        adj_matrix_np_verify = graph_to_adj_matrix_numpy(G_verify)
        
        verification_results = {}
        
        # Run traditional PageRank
        print("Running traditional PageRank for verification...")
        traditional_result = PageRank.traditional_pagerank_cpu(
            adj_list_verify, 
            damping=args.damping, 
            max_iterations=args.max_iters, 
            tol=args.tolerance
        )
        verification_results['Traditional_PageRank'] = traditional_result
        
        # Run multiprocessing PageRank with a reasonable process count
        if not args.skip_mp:
            process_count = min(4, max_cores)
            print(f"Running multiprocessing PageRank with {process_count} processes for verification...")
            try:
                mp_result = PageRank.traditional_pagerank_multiprocessing(
                    adj_list_verify,
                    damping=args.damping,
                    max_iterations=args.max_iters,
                    tol=args.tolerance,
                    num_processes=process_count
                )
                verification_results[f'MP_PageRank_{process_count}processes'] = mp_result
            except Exception as e:
                print(f"Error in multiprocessing PageRank verification: {e}")
        
        # Run linear algebra PageRank on CPU
        print("Running linear algebra PageRank on CPU for verification...")
        try:
            la_cpu_result = PageRank.la_pagerank_cpu(
                adj_matrix_np_verify, 
                damping=args.damping, 
                max_iterations=args.max_iters, 
                tol=args.tolerance
            )
            verification_results['LA_PageRank_CPU'] = la_cpu_result
        except Exception as e:
            print(f"Error in LA PageRank verification: {e}")
        
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
        
        # Run traditional PageRank
        print("Running traditional PageRank...")
        try:
            # For timing
            total_time = 0
            iterations_sum = 0
            for i in range(args.runs):
                start_time = time.time()
                ranks, iterations = PageRank.traditional_pagerank_cpu(
                    adj_list,
                    damping=args.damping,
                    max_iterations=args.max_iters,
                    tol=args.tolerance
                )
                end_time = time.time()
                total_time += (end_time - start_time)
                iterations_sum += iterations
            
            avg_time = total_time / args.runs
            avg_iterations = iterations_sum / args.runs
            
            trad_result = {
                'avg_time': avg_time,
                'iterations': avg_iterations,
                'size': size
            }
            
            benchmark.add_result('Traditional_PageRank', args.graph_type, size, trad_result)
            print(f"Average time: {trad_result['avg_time']:.6f} seconds, Iterations: {trad_result['iterations']:.1f}")
        except Exception as e:
            print(f"Error in traditional PageRank: {e}")
        
        # Run multiprocessing-based PageRank
        if not args.skip_mp:
            print("Running multiprocessing PageRank...")
            
            # Test with different process counts
            for process_count in args.processes:
                if process_count <= max_cores:
                    print(f"  With {process_count} processes:")
                    
                    try:
                        # For timing
                        total_time = 0
                        iterations_sum = 0
                        for i in range(args.runs):
                            start_time = time.time()
                            ranks, iterations = PageRank.traditional_pagerank_multiprocessing(
                                adj_list,
                                damping=args.damping,
                                max_iterations=args.max_iters,
                                tol=args.tolerance,
                                num_processes=process_count
                            )
                            end_time = time.time()
                            total_time += (end_time - start_time)
                            iterations_sum += iterations
                        
                        avg_time = total_time / args.runs
                        avg_iterations = iterations_sum / args.runs
                        
                        mp_result = {
                            'avg_time': avg_time,
                            'iterations': avg_iterations,
                            'size': size
                        }
                        
                        benchmark.add_result(f'MP_PageRank_{process_count}processes', args.graph_type, size, mp_result)
                        print(f"  Multiprocessing PageRank: {mp_result['avg_time']:.6f} seconds, Iterations: {mp_result['iterations']:.1f}")
                    except Exception as e:
                        print(f"  Error in multiprocessing PageRank with {process_count} processes: {e}")
        
        # Run linear algebra PageRank on CPU
        print("Running linear algebra PageRank on CPU...")
        try:
            # For timing
            total_time = 0
            iterations_sum = 0
            for i in range(args.runs):
                start_time = time.time()
                ranks, iterations = PageRank.la_pagerank_cpu(
                    adj_matrix_np,
                    damping=args.damping,
                    max_iterations=args.max_iters,
                    tol=args.tolerance
                )
                end_time = time.time()
                total_time += (end_time - start_time)
                iterations_sum += iterations
            
            avg_time = total_time / args.runs
            avg_iterations = iterations_sum / args.runs
            
            la_cpu_result = {
                'avg_time': avg_time,
                'iterations': avg_iterations,
                'size': size
            }
            
            benchmark.add_result('LA_PageRank_CPU', args.graph_type, size, la_cpu_result)
            print(f"Average time: {la_cpu_result['avg_time']:.6f} seconds, Iterations: {la_cpu_result['iterations']:.1f}")
        except Exception as e:
            print(f"Error in LA PageRank CPU: {e}")
        
        # Run linear algebra PageRank on GPU if requested
        if args.gpu:
            print("Running linear algebra PageRank on GPU (dense)...")
            try:
                # For timing
                total_time = 0
                iterations_sum = 0
                for i in range(args.runs):
                    start_time = time.time()
                    ranks, iterations = PageRank.la_pagerank_gpu(
                        adj_matrix_torch,
                        damping=args.damping,
                        max_iterations=args.max_iters,
                        tol=args.tolerance
                    )
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    iterations_sum += iterations
                
                avg_time = total_time / args.runs
                avg_iterations = iterations_sum / args.runs
                
                la_gpu_result = {
                    'avg_time': avg_time,
                    'iterations': avg_iterations,
                    'size': size
                }
                
                benchmark.add_result('LA_PageRank_GPU_Dense', args.graph_type, size, la_gpu_result)
                print(f"Average time: {la_gpu_result['avg_time']:.6f} seconds, Iterations: {la_gpu_result['iterations']:.1f}")
            except Exception as e:
                print(f"Error in LA PageRank GPU (dense): {e}")
            
            print("Running linear algebra PageRank on GPU (sparse)...")
            try:
                # For timing
                total_time = 0
                iterations_sum = 0
                for i in range(args.runs):
                    start_time = time.time()
                    ranks, iterations = PageRank.la_pagerank_sparse_gpu(
                        adj_matrix_sparse,
                        damping=args.damping,
                        max_iterations=args.max_iters,
                        tol=args.tolerance
                    )
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    iterations_sum += iterations
                
                avg_time = total_time / args.runs
                avg_iterations = iterations_sum / args.runs
                
                la_sparse_result = {
                    'avg_time': avg_time,
                    'iterations': avg_iterations,
                    'size': size
                }
                
                benchmark.add_result('LA_PageRank_GPU_Sparse', args.graph_type, size, la_sparse_result)
                print(f"Average time: {la_sparse_result['avg_time']:.6f} seconds, Iterations: {la_sparse_result['iterations']:.1f}")
            except Exception as e:
                print(f"Error in LA PageRank GPU (sparse): {e}")
    
    # Save results
    benchmark.save_results()
    benchmark.print_results()
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating performance comparison plot...")
        benchmark.plot_comparison(
            args.graph_type,
            save_file=f'pagerank_{args.graph_type}_performance.png'
        )
        
        print("Generating speedup comparison plot...")
        benchmark.plot_speedup(
            'Traditional_PageRank',
            args.graph_type,
            save_file=f'pagerank_{args.graph_type}_speedup.png'
        )
        
        # Generate multiprocessing scaling plot if applicable
        if not args.skip_mp:
            print("Generating multiprocessing scaling plot...")
            mp_results = {}
            for process_count in args.processes:
                mp_results[f'MP_PageRank_{process_count}processes'] = benchmark.get_results(f'MP_PageRank_{process_count}processes', args.graph_type)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            for key, results in mp_results.items():
                if results:  # Only plot if we have results
                    sizes = [r['size'] for r in results]
                    times = [r['avg_time'] for r in results]
                    plt.plot(sizes, times, marker='o', label=key)
            
            plt.xlabel('Graph Size (nodes)')
            plt.ylabel('Time (seconds)')
            plt.title(f'PageRank Multiprocessing Scaling ({args.graph_type} graph)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'pagerank_{args.graph_type}_mp_scaling.png')
            plt.close()
        
        # Generate algorithmic comparison plot
        print("Generating algorithm comparison plot...")
        algorithm_results = {
            'Traditional_PageRank': benchmark.get_results('Traditional_PageRank', args.graph_type),
            'LA_PageRank_CPU': benchmark.get_results('LA_PageRank_CPU', args.graph_type)
        }
        
        # Add best multiprocessing result
        if not args.skip_mp:
            best_process_count = max(args.processes)
            mp_key = f'MP_PageRank_{best_process_count}processes'
            mp_results = benchmark.get_results(mp_key, args.graph_type)
            if mp_results:
                algorithm_results[mp_key] = mp_results
        
        if args.gpu:
            algorithm_results['LA_PageRank_GPU_Dense'] = benchmark.get_results('LA_PageRank_GPU_Dense', args.graph_type)
            algorithm_results['LA_PageRank_GPU_Sparse'] = benchmark.get_results('LA_PageRank_GPU_Sparse', args.graph_type)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for key, results in algorithm_results.items():
            if results:  # Only plot if we have results
                sizes = [r['size'] for r in results]
                times = [r['avg_time'] for r in results]
                plt.plot(sizes, times, marker='o', linewidth=2, label=key)
        
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Time (seconds)')
        plt.title(f'PageRank Algorithm Comparison ({args.graph_type} graph)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Log scale often helps visualize large performance differences
        plt.savefig(f'pagerank_{args.graph_type}_algorithm_comparison.png')
        plt.close()
        
        # Generate convergence iterations plot
        print("Generating convergence iterations plot...")
        convergence_results = {
            'Traditional_PageRank': benchmark.get_results('Traditional_PageRank', args.graph_type),
            'LA_PageRank_CPU': benchmark.get_results('LA_PageRank_CPU', args.graph_type)
        }
        
        # Add multiprocessing results
        if not args.skip_mp:
            best_process_count = max(args.processes)
            mp_key = f'MP_PageRank_{best_process_count}processes'
            mp_results = benchmark.get_results(mp_key, args.graph_type)
            if mp_results:
                convergence_results[mp_key] = mp_results
        
        if args.gpu:
            convergence_results['LA_PageRank_GPU_Dense'] = benchmark.get_results('LA_PageRank_GPU_Dense', args.graph_type)
            convergence_results['LA_PageRank_GPU_Sparse'] = benchmark.get_results('LA_PageRank_GPU_Sparse', args.graph_type)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for key, results in convergence_results.items():
            if results:  # Only plot if we have results with iteration data
                sizes = [r['size'] for r in results]
                iterations = [r.get('iterations', 0) for r in results]  # Use get() with default value
                plt.plot(sizes, iterations, marker='o', linewidth=2, label=key)
        
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Iterations to Convergence')
        plt.title(f'PageRank Convergence Iterations ({args.graph_type} graph)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'pagerank_{args.graph_type}_convergence.png')
        plt.close()
        
        # Generate parallel approaches comparison plot
        print("Generating parallel approaches comparison plot...")
        parallel_results = {}
        
        # Add best multiprocessing result
        if not args.skip_mp:
            best_process_count = max(args.processes)
            mp_key = f'MP_{best_process_count}processes'
            mp_results = benchmark.get_results(f'MP_PageRank_{best_process_count}processes', args.graph_type)
            if mp_results:
                parallel_results[mp_key] = mp_results
        
        # Add traditional for reference
        parallel_results['Traditional'] = benchmark.get_results('Traditional_PageRank', args.graph_type)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for key, results in parallel_results.items():
            if results:  # Only plot if we have results
                sizes = [r['size'] for r in results]
                times = [r['avg_time'] for r in results]
                plt.plot(sizes, times, marker='o', linewidth=2, label=key)
        
        plt.xlabel('Graph Size (nodes)')
        plt.ylabel('Time (seconds)')
        plt.title(f'PageRank Parallel Approaches Comparison ({args.graph_type} graph)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  
        plt.savefig(f'pagerank_{args.graph_type}_parallel_comparison.png')
        plt.close()

if __name__ == '__main__':
    main()