#!/usr/bin/env python3
"""
Script to run BFS benchmarks comparing traditional CPU, single-GPU sparse matrix,
and multi-GPU implementations with strong and weak scaling
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import gc
import psutil
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
from src.algorithms.multi_gpu_bfs import MultiGPUBFS
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

def run_strong_scaling_test(base_size, num_gpus_list, graph_type='scale-free', seed=42, runs=3):
    """
    Run strong scaling test (fixed problem size, varying number of GPUs)
    
    Parameters:
    -----------
    base_size : int
        Fixed graph size for all tests
    num_gpus_list : list
        List of GPU counts to test
    graph_type : str
        Type of graph to generate
    seed : int
        Random seed
    runs : int
        Number of runs for each configuration
        
    Returns:
    --------
    dict
        Results for each configuration
    """
    print(f"\n{'='*80}")
    print(f"STRONG SCALING TEST: Fixed size {base_size} nodes, varying number of GPUs")
    print(f"{'='*80}")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define graph generation function based on type
    if graph_type == 'random':
        def generate_graph(n):
            p = 10 / (n - 1)  # Avg degree of 10
            return generate_random_graph(n, p, seed=seed)
    elif graph_type == 'scale-free':
        def generate_graph(n):
            return generate_scale_free_graph(n, m=5, seed=seed)
    elif graph_type == 'small-world':
        def generate_graph(n):
            return generate_small_world_graph(n, k=10, p=0.1, seed=seed)
    
    # Generate graph of fixed size
    print(f"\nGenerating {graph_type} graph with {base_size} nodes...")
    G = generate_graph(base_size)
    print_simplified_graph_stats(G)
    
    # Choose a start node (node with highest degree)
    start_node = max(G.degree(), key=lambda x: x[1])[0]
    print(f"Start node: {start_node} with degree {G.degree(start_node)}")
    
    # Convert graph to adjacency list for traditional BFS
    print("Converting graph to adjacency list...")
    adj_list = graph_to_adj_list(G)
    
    # Convert graph to sparse matrix for GPU methods
    print("Converting graph to sparse matrix for GPU implementations...")
    adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device='cuda:0')
    print("Sparse matrix conversion complete")
    
    # Print memory usage
    print("\nMemory usage after conversions:")
    print_memory_usage()
    
    # Store results
    results = {}
    
    # Run traditional CPU BFS
    print("\nRunning traditional BFS on CPU...")
    try:
        traditional_result, _ = run_test(
            BFS.traditional_bfs_cpu,
            adj_list,
            start_node,
            n_runs=runs
        )
        results['Traditional_CPU'] = traditional_result
        print(f"Average time: {traditional_result['avg_time']:.6f} seconds")
    except Exception as e:
        print(f"Error in traditional BFS: {e}")
    
    # Run single-GPU sparse BFS
    print("\nRunning optimized sparse BFS on single GPU...")
    try:
        gpu_opt_result, _ = run_test(
            BFS.la_bfs_sparse_gpu_optimized_v2_turbo,
            adj_matrix_sparse,
            start_node,
            n_runs=runs
        )
        results['Single_GPU_Optimized'] = gpu_opt_result
        print(f"Average time: {gpu_opt_result['avg_time']:.6f} seconds")
    except Exception as e:
        print(f"Error in single-GPU optimized BFS: {e}")
    
    # Run multi-GPU BFS for each GPU count
    for num_gpus in num_gpus_list:
        if num_gpus > torch.cuda.device_count():
            print(f"\nSkipping {num_gpus} GPUs test (only {torch.cuda.device_count()} available)")
            continue
            
        print(f"\nRunning multi-GPU BFS with {num_gpus} GPUs...")
        try:
            # Clear memory before multi-GPU test
            torch.cuda.empty_cache()
            gc.collect()
            
            # Run multi-GPU version
            multi_gpu_result, _ = run_test(
                MultiGPUBFS.la_bfs_multi_gpu_v2,
                adj_matrix_sparse,
                start_node,
                num_gpus=num_gpus,
                n_runs=runs
            )
            results[f'Multi_GPU_{num_gpus}'] = multi_gpu_result
            print(f"Average time: {multi_gpu_result['avg_time']:.6f} seconds")
            
            # Calculate speedup vs traditional
            if 'Traditional_CPU' in results:
                cpu_time = results['Traditional_CPU']['avg_time']
                speedup_vs_cpu = cpu_time / multi_gpu_result['avg_time']
                print(f"Speedup vs traditional CPU: {speedup_vs_cpu:.2f}x")
            
            # Calculate speedup vs single GPU
            if 'Single_GPU_Optimized' in results:
                single_gpu_time = results['Single_GPU_Optimized']['avg_time']
                speedup_vs_single = single_gpu_time / multi_gpu_result['avg_time']
                print(f"Speedup vs single GPU: {speedup_vs_single:.2f}x")
            
            # Calculate parallel efficiency
            if 'Single_GPU_Optimized' in results and num_gpus > 1:
                single_gpu_time = results['Single_GPU_Optimized']['avg_time']
                parallel_efficiency = (single_gpu_time / multi_gpu_result['avg_time']) / num_gpus
                print(f"Parallel efficiency: {parallel_efficiency * 100:.2f}%")
                
        except Exception as e:
            print(f"Error in multi-GPU BFS with {num_gpus} GPUs: {e}")
    
    # Clean up
    del adj_list, adj_matrix_sparse, G
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def run_weak_scaling_test(base_size, num_gpus_list, graph_type='scale-free', seed=42, runs=3):
    """
    Run weak scaling test (problem size grows with number of GPUs)
    
    Parameters:
    -----------
    base_size : int
        Base graph size for single GPU, scaled for multiple GPUs
    num_gpus_list : list
        List of GPU counts to test
    graph_type : str
        Type of graph to generate
    seed : int
        Random seed
    runs : int
        Number of runs for each configuration
        
    Returns:
    --------
    dict
        Results for each configuration
    """
    print(f"\n{'='*80}")
    print(f"WEAK SCALING TEST: Size grows with number of GPUs, starting at {base_size}")
    print(f"{'='*80}")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define graph generation function based on type
    if graph_type == 'random':
        def generate_graph(n):
            p = 10 / (n - 1)  # Avg degree of 10
            return generate_random_graph(n, p, seed=seed)
    elif graph_type == 'scale-free':
        def generate_graph(n):
            return generate_scale_free_graph(n, m=5, seed=seed)
    elif graph_type == 'small-world':
        def generate_graph(n):
            return generate_small_world_graph(n, k=10, p=0.1, seed=seed)
    
    # Store results
    results = {}
    
    # Run for each GPU count
    for num_gpus in num_gpus_list:
        if num_gpus > torch.cuda.device_count():
            print(f"\nSkipping {num_gpus} GPUs test (only {torch.cuda.device_count()} available)")
            continue
        
        # Calculate graph size based on number of GPUs (linear scaling)
        graph_size = base_size * num_gpus
        
        print(f"\n{'-'*60}")
        print(f"Testing with {num_gpus} GPUs and graph size {graph_size}")
        print(f"{'-'*60}")
        
        # Generate graph
        print(f"Generating {graph_type} graph with {graph_size} nodes...")
        G = generate_graph(graph_size)
        print_simplified_graph_stats(G)
        
        # Choose a start node (node with highest degree)
        start_node = max(G.degree(), key=lambda x: x[1])[0]
        print(f"Start node: {start_node} with degree {G.degree(start_node)}")
        
        # Convert graph to adjacency list for traditional BFS
        print("Converting graph to adjacency list...")
        adj_list = graph_to_adj_list(G)
        
        # Convert graph to sparse matrix for GPU methods
        print("Converting graph to sparse matrix for GPU implementations...")
        adj_matrix_sparse = graph_to_sparse_adj_matrix_torch(G, device='cuda:0')
        print("Sparse matrix conversion complete")
        
        # Print memory usage
        print("\nMemory usage after conversions:")
        print_memory_usage()
        
        # Store results for this configuration
        config_results = {}
        
        # Run traditional CPU BFS if graph is not too large
        if graph_size <= 100000:  # Limit for CPU tests
            print("\nRunning traditional BFS on CPU...")
            try:
                traditional_result, _ = run_test(
                    BFS.traditional_bfs_cpu,
                    adj_list,
                    start_node,
                    n_runs=runs
                )
                config_results['Traditional_CPU'] = traditional_result
                print(f"Average time: {traditional_result['avg_time']:.6f} seconds")
            except Exception as e:
                print(f"Error in traditional BFS: {e}")
        else:
            print("\nSkipping traditional BFS on CPU (graph too large)")
        
        # Run single-GPU sparse BFS
        print("\nRunning optimized sparse BFS on single GPU...")
        try:
            gpu_opt_result, _ = run_test(
                BFS.la_bfs_sparse_gpu_optimized_v2_turbo,
                adj_matrix_sparse,
                start_node,
                n_runs=runs
            )
            config_results['Single_GPU_Optimized'] = gpu_opt_result
            print(f"Average time: {gpu_opt_result['avg_time']:.6f} seconds")
        except Exception as e:
            print(f"Error in single-GPU optimized BFS: {e}")
        
        # Run multi-GPU BFS
        print(f"\nRunning multi-GPU BFS with {num_gpus} GPUs...")
        try:
            # Clear memory before multi-GPU test
            torch.cuda.empty_cache()
            gc.collect()
            
            # Run multi-GPU version
            multi_gpu_result, _ = run_test(
                MultiGPUBFS.la_bfs_multi_gpu_v2,
                adj_matrix_sparse,
                start_node,
                num_gpus=num_gpus,
                n_runs=runs
            )
            config_results['Multi_GPU'] = multi_gpu_result
            print(f"Average time: {multi_gpu_result['avg_time']:.6f} seconds")
            
            # Calculate speedup vs traditional if available
            if 'Traditional_CPU' in config_results:
                cpu_time = config_results['Traditional_CPU']['avg_time']
                speedup_vs_cpu = cpu_time / multi_gpu_result['avg_time']
                print(f"Speedup vs traditional CPU: {speedup_vs_cpu:.2f}x")
            
            # Calculate speedup vs single GPU
            if 'Single_GPU_Optimized' in config_results:
                single_gpu_time = config_results['Single_GPU_Optimized']['avg_time']
                speedup_vs_single = single_gpu_time / multi_gpu_result['avg_time']
                print(f"Speedup vs single GPU: {speedup_vs_single:.2f}x")
            
        except Exception as e:
            print(f"Error in multi-GPU BFS: {e}")
        
        # Store results for this GPU count
        results[num_gpus] = config_results
        
        # Clean up
        del adj_list, adj_matrix_sparse, G
        gc.collect()
        torch.cuda.empty_cache()
    
    return results

def plot_strong_scaling_results(results, output_dir=None):
    """Generate plots for strong scaling results"""
    # Get GPU counts and execution times
    gpu_counts = []
    cpu_times = []
    single_gpu_times = []
    multi_gpu_times = []
    speedups_vs_cpu = []
    speedups_vs_single = []
    parallel_efficiencies = []
    
    # Extract data
    if 'Traditional_CPU' in results:
        cpu_time = results['Traditional_CPU']['avg_time']
    else:
        cpu_time = None
        
    if 'Single_GPU_Optimized' in results:
        single_gpu_time = results['Single_GPU_Optimized']['avg_time']
    else:
        single_gpu_time = None
    
    # Extract multi-GPU data
    for key in results.keys():
        if key.startswith('Multi_GPU_'):
            num_gpus = int(key.split('_')[-1])
            time = results[key]['avg_time']
            
            gpu_counts.append(num_gpus)
            multi_gpu_times.append(time)
            
            if cpu_time:
                speedup_vs_cpu = cpu_time / time
                speedups_vs_cpu.append(speedup_vs_cpu)
            
            if single_gpu_time:
                speedup_vs_single = single_gpu_time / time
                speedups_vs_single.append(speedup_vs_single)
                
                if num_gpus > 1:
                    parallel_efficiency = speedup_vs_single / num_gpus
                    parallel_efficiencies.append(parallel_efficiency)
    
    # Sort data by GPU count
    sorted_data = sorted(zip(gpu_counts, multi_gpu_times), key=lambda x: x[0])
    gpu_counts = [x[0] for x in sorted_data]
    multi_gpu_times = [x[1] for x in sorted_data]
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Execution Times
    plt.subplot(2, 2, 1)
    plt.plot(gpu_counts, multi_gpu_times, 'o-', label='Multi-GPU')
    if single_gpu_time:
        plt.axhline(y=single_gpu_time, linestyle='--', color='r', label='Single GPU')
    if cpu_time:
        plt.axhline(y=cpu_time, linestyle='--', color='g', label='CPU')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Execution Time (s)')
    plt.title('BFS Execution Time vs. Number of GPUs')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Speedup vs. Single GPU
    if single_gpu_time:
        plt.subplot(2, 2, 2)
        sorted_speedups = [x/y for x, y in zip([single_gpu_time]*len(gpu_counts), multi_gpu_times)]
        plt.plot(gpu_counts, sorted_speedups, 'o-', label='Actual')
        plt.plot(gpu_counts, gpu_counts, '--', label='Ideal Linear Speedup')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Speedup vs. Single GPU')
        plt.title('Speedup vs. Single GPU')
        plt.grid(True)
        plt.legend()
    
    # Plot 3: Parallel Efficiency
    if single_gpu_time:
        plt.subplot(2, 2, 3)
        sorted_efficiencies = [x/(y*z) for x, y, z in zip([single_gpu_time]*len(gpu_counts), multi_gpu_times, gpu_counts)]
        plt.plot(gpu_counts, sorted_efficiencies, 'o-')
        plt.axhline(y=1.0, linestyle='--', color='r', label='Ideal')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Parallel Efficiency')
        plt.title('Parallel Efficiency')
        plt.grid(True)
        plt.ylim(0, 1.2)
    
    # Plot 4: Speedup vs. CPU
    if cpu_time:
        plt.subplot(2, 2, 4)
        sorted_cpu_speedups = [x/y for x, y in zip([cpu_time]*len(gpu_counts), multi_gpu_times)]
        plt.plot(gpu_counts, sorted_cpu_speedups, 'o-')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Speedup vs. CPU')
        plt.title('Speedup vs. CPU')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'bfs_strong_scaling.png'), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_weak_scaling_results(results, output_dir=None):
    """Generate plots for weak scaling results"""
    # Extract data
    gpu_counts = sorted(list(results.keys()))
    multi_gpu_times = []
    single_gpu_times = []
    cpu_times = []
    
    for num_gpus in gpu_counts:
        if 'Multi_GPU' in results[num_gpus]:
            multi_gpu_times.append(results[num_gpus]['Multi_GPU']['avg_time'])
        else:
            multi_gpu_times.append(None)
            
        if 'Single_GPU_Optimized' in results[num_gpus]:
            single_gpu_times.append(results[num_gpus]['Single_GPU_Optimized']['avg_time'])
        else:
            single_gpu_times.append(None)
            
        if 'Traditional_CPU' in results[num_gpus]:
            cpu_times.append(results[num_gpus]['Traditional_CPU']['avg_time'])
        else:
            cpu_times.append(None)
    
    # Filter out None values
    valid_multi_gpu = [(g, t) for g, t in zip(gpu_counts, multi_gpu_times) if t is not None]
    valid_single_gpu = [(g, t) for g, t in zip(gpu_counts, single_gpu_times) if t is not None]
    valid_cpu = [(g, t) for g, t in zip(gpu_counts, cpu_times) if t is not None]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot execution times
    if valid_multi_gpu:
        plt.plot([x[0] for x in valid_multi_gpu], [x[1] for x in valid_multi_gpu], 'o-', label='Multi-GPU')
    if valid_single_gpu:
        plt.plot([x[0] for x in valid_single_gpu], [x[1] for x in valid_single_gpu], 's-', label='Single GPU')
    if valid_cpu:
        plt.plot([x[0] for x in valid_cpu], [x[1] for x in valid_cpu], '^-', label='CPU')
    
    plt.xlabel('Number of GPUs (Problem Size = Base Size × Number of GPUs)')
    plt.ylabel('Execution Time (s)')
    plt.title('BFS Weak Scaling: Execution Time vs. Number of GPUs')
    plt.grid(True)
    plt.legend()
    
    # Save plot if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'bfs_weak_scaling.png'), dpi=300)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run BFS benchmarks with multi-GPU scaling')
    parser.add_argument('--base-size', type=int, default=2000000, 
                        help='Base graph size for tests')
    parser.add_argument('--graph-type', type=str, choices=['random', 'scale-free', 'small-world'], 
                        default='scale-free', help='Type of graph to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each benchmark')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results and plots')
    parser.add_argument('--strong-scaling', action='store_true', 
                        help='Run strong scaling test')
    parser.add_argument('--weak-scaling', action='store_true',
                        help='Run weak scaling test')
    parser.add_argument('--gpu-counts', type=int, nargs='+', 
                        default=[1, 2, 3, 4], 
                        help='GPU counts to test')
    
    args = parser.parse_args()
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("Error: GPU not available. This script requires GPU support.")
        return
    
    # Print available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Filter GPU counts based on available GPUs
    gpu_counts = [count for count in args.gpu_counts if count <= num_gpus]
    if not gpu_counts:
        print(f"Error: No valid GPU counts specified. Available GPUs: {num_gpus}")
        return
    
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f'bfs_{args.graph_type}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run strong scaling test
    if args.strong_scaling:
        strong_results = run_strong_scaling_test(
            args.base_size, 
            gpu_counts, 
            args.graph_type, 
            args.seed,
            args.runs
        )
        results['strong_scaling'] = strong_results
        
        # Plot results
        plot_strong_scaling_results(strong_results, output_dir)
        
        # Save results to file
        with open(os.path.join(output_dir, 'strong_scaling_results.json'), 'w') as f:
            # Convert NumPy values to Python types for JSON serialization
            json_results = {}
            for k, v in strong_results.items():
                json_results[k] = {kk: float(vv) if isinstance(vv, (np.float32, np.float64)) else vv 
                                  for kk, vv in v.items()}
            json.dump(json_results, f, indent=2)
    
    # Run weak scaling test
    if args.weak_scaling:
        weak_results = run_weak_scaling_test(
            args.base_size, 
            gpu_counts, 
            args.graph_type, 
            args.seed,
            args.runs
        )
        results['weak_scaling'] = weak_results
        
        # Plot results
        plot_weak_scaling_results(weak_results, output_dir)
        
        # Save results to file
        with open(os.path.join(output_dir, 'weak_scaling_results.json'), 'w') as f:
            # Convert NumPy values to Python types for JSON serialization
            json_results = {}
            for k, v in weak_results.items():
                json_results[k] = {kk: {kkk: float(vvv) if isinstance(vvv, (np.float32, np.float64)) else vvv 
                                        for kkk, vvv in vv.items()} 
                                  for kk, vv in v.items()}
            json.dump(json_results, f, indent=2)
    
    print(f"\nBenchmark complete! Results saved to {output_dir}")

if __name__ == '__main__':
    main()