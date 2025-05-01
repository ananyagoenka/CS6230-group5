import time
import numpy as np
import torch
import cupy as cp
import matplotlib.pyplot as plt
from pathlib import Path

def time_function(func, *args, n_runs=10, warmup=3, **kwargs):
    """
    Time a function execution.
    
    Args:
        func: Function to time
        *args: Arguments to pass to the function
        n_runs (int): Number of runs to average over
        warmup (int): Number of warmup runs
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        float: Average execution time in seconds
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def memory_usage(func, *args, **kwargs):
    """
    Measure peak memory usage of a function.
    
    Args:
        func: Function to measure
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        float: Peak memory usage in MB
    """
    # Record initial memory usage
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Record peak memory usage
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return peak - initial, result

def speedup(baseline_time, optimized_time):
    """
    Calculate speedup.
    
    Args:
        baseline_time (float): Baseline execution time
        optimized_time (float): Optimized execution time
        
    Returns:
        float: Speedup factor
    """
    return baseline_time / optimized_time

def plot_scaling(sizes, times, labels, title, output_file=None):
    """
    Plot scaling results.
    
    Args:
        sizes (list): Problem sizes
        times (list): List of time measurements for each implementation
        labels (list): List of labels for each implementation
        title (str): Plot title
        output_file (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, t in enumerate(times):
        plt.plot(sizes, t, marker='o', label=labels[i])
    
    plt.xlabel('Problem Size')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def plot_speedup(sizes, baseline_times, optimized_times, labels, title, output_file=None):
    """
    Plot speedup results.
    
    Args:
        sizes (list): Problem sizes
        baseline_times (list): Baseline execution times
        optimized_times (list): List of optimized execution times for each implementation
        labels (list): List of labels for each implementation
        title (str): Plot title
        output_file (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, t in enumerate(optimized_times):
        speedups = [baseline_times[j] / t[j] for j in range(len(sizes))]
        plt.plot(sizes, speedups, marker='o', label=labels[i])
    
    plt.xlabel('Problem Size')
    plt.ylabel('Speedup')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def save_benchmark_results(results, output_file):
    """
    Save benchmark results to file.
    
    Args:
        results (dict): Benchmark results
        output_file (str): Path to save the results
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    np.save(output_file, results)

def load_benchmark_results(input_file):
    """
    Load benchmark results from file.
    
    Args:
        input_file (str): Path to the results file
        
    Returns:
        dict: Benchmark results
    """
    return np.load(input_file, allow_pickle=True).item()