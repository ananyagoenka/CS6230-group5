#!/usr/bin/env python3
"""
Script to run all benchmarks (BFS and PageRank)
"""

import os
import sys
import argparse
import subprocess
import time
import multiprocessing
import numpy as np

def run_command(cmd, verbose=True):
    """
    Run a shell command and optionally print its output
    
    Parameters:
    -----------
    cmd : str
        Command to run
    verbose : bool
        Whether to print command output
    """
    if verbose:
        print(f"Running command: {cmd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        if verbose:
            print(line.strip())
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run all benchmarks')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 5000, 10000], 
                        help='Graph sizes to benchmark')
    parser.add_argument('--graph-types', type=str, nargs='+', 
                        choices=['random', 'scale-free', 'small-world'], 
                        default=['scale-free'], 
                        help='Types of graphs to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for each benchmark')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--parallel', action='store_true', 
                        help='Run different graph types in parallel')
    parser.add_argument('--verbose', action='store_true', help='Print command output')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Construct base commands
    sizes_str = ' '.join(map(str, args.sizes))
    gpu_flag = '--gpu' if args.gpu else ''
    plot_flag = '--plot' if args.plot else ''
    
    # Generate commands for each algorithm and graph type
    commands = []
    
    for graph_type in args.graph_types:
        # BFS benchmark
        bfs_cmd = (
            f"python scripts/run_bfs.py "
            f"--sizes {sizes_str} "
            f"--graph-type {graph_type} "
            f"--seed {args.seed} "
            f"--runs {args.runs} "
            f"--save-dir {args.save_dir} "
            f"{gpu_flag} {plot_flag}"
        )
        commands.append(bfs_cmd)
        
        # PageRank benchmark
        pr_cmd = (
            f"python scripts/run_pagerank.py "
            f"--sizes {sizes_str} "
            f"--graph-type {graph_type} "
            f"--seed {args.seed} "
            f"--runs {args.runs} "
            f"--save-dir {args.save_dir} "
            f"{gpu_flag} {plot_flag}"
        )
        commands.append(pr_cmd)
    
    # Run commands
    if args.parallel and len(commands) > 1:
        print(f"Running {len(commands)} commands in parallel...")
        
        # Create a pool of workers
        pool = multiprocessing.Pool(min(len(commands), multiprocessing.cpu_count()))
        
        # Map commands to workers
        results = pool.map(lambda cmd: run_command(cmd, args.verbose), commands)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Check results
        if all(results):
            print("\nAll benchmarks completed successfully.")
        else:
            print("\nSome benchmarks failed. Check the output for details.")
    else:
        print(f"Running {len(commands)} commands sequentially...")
        
        # Run commands sequentially
        for cmd in commands:
            if not run_command(cmd, args.verbose):
                print("Benchmark failed. Stopping execution.")
                return
        
        print("\nAll benchmarks completed successfully.")

if __name__ == '__main__':
    main()