import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.bfs import MatrixBFS
from src.algorithms.pagerank import MatrixPageRank
from src.algorithms.connected_components import MatrixConnectedComponents
from src.utils.graph_io import load_sparse_matrix
from src.traditional.bfs import NetworkXBFS
from src.traditional.pagerank import NetworkXPageRank
from src.traditional.connected_components import NetworkXCC
from benchmarks.benchmark_utils import time_function, save_benchmark_results, plot_scaling, plot_speedup

def run_strong_scaling_bfs(adj_matrices, source_nodes, num_nodes, devices, output_dir="results/strong_scaling"):
    """
    Run strong scaling benchmark for BFS.
    
    Args:
        adj_matrices (list): List of adjacency matrices
        source_nodes (list): List of source nodes for each graph
        num_nodes (list): List of number of nodes for each graph
        devices (list): List of devices to test on
        output_dir (str): Directory to save results
    """
    results = {
        "graph_sizes": num_nodes,
        "devices": devices,
        "matrix_bfs_times": [],
        "matrix_bfs_std": [],
        "networkx_bfs_times": [],
        "networkx_bfs_std": []
    }
    
    for i, adj_matrix in enumerate(adj_matrices):
        source = source_nodes[i]
        n = num_nodes[i]
        
        matrix_times = []
        matrix_stds = []
        
        for device in devices:
            # Matrix BFS
            bfs = MatrixBFS(device=device)
            bfs.preprocess(adj_matrix)
            
            # Time execution
            mean_time, std_time = time_function(bfs.run, source, n, n_runs=5)
            matrix_times.append(mean_time)
            matrix_stds.append(std_time)
            
            print(f"Graph size: {n}, Device: {device}, Matrix BFS time: {mean_time:.6f} ± {std_time:.6f} s")
        
        results["matrix_bfs_times"].append(matrix_times)
        results["matrix_bfs_std"].append(matrix_stds)
        
        # NetworkX BFS (only on CPU)
        nx_bfs = NetworkXBFS()
        nx_bfs.preprocess(adj_matrix)
        
        # Time execution
        mean_time, std_time = time_function(nx_bfs.run, source, n_runs=5)
        results["networkx_bfs_times"].append(mean_time)
        results["networkx_bfs_std"].append(std_time)
        
        print(f"Graph size: {n}, NetworkX BFS time: {mean_time:.6f} ± {std_time:.6f} s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_benchmark_results(results, f"{output_dir}/bfs_strong_scaling.npy")
    
    # Plot results
    plot_scaling(
        num_nodes, 
        [results["networkx_bfs_times"]] + [
            [results["matrix_bfs_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        ["NetworkX (CPU)"] + [f"Matrix BFS ({device})" for device in devices],
        "BFS Strong Scaling",
        f"{output_dir}/bfs_strong_scaling.png"
    )
    
    # Plot speedup
    plot_speedup(
        num_nodes,
        results["networkx_bfs_times"],
        [
            [results["matrix_bfs_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        [f"Matrix BFS ({device})" for device in devices],
        "BFS Speedup vs. NetworkX",
        f"{output_dir}/bfs_speedup.png"
    )

def run_strong_scaling_pagerank(adj_matrices, num_nodes, devices, output_dir="results/strong_scaling"):
    """
    Run strong scaling benchmark for PageRank.
    
    Args:
        adj_matrices (list): List of adjacency matrices
        num_nodes (list): List of number of nodes for each graph
        devices (list): List of devices to test on
        output_dir (str): Directory to save results
    """
    results = {
        "graph_sizes": num_nodes,
        "devices": devices,
        "matrix_pr_times": [],
        "matrix_pr_std": [],
        "networkx_pr_times": [],
        "networkx_pr_std": []
    }
    
    for i, adj_matrix in enumerate(adj_matrices):
        n = num_nodes[i]
        
        matrix_times = []
        matrix_stds = []
        
        for device in devices:
            # Matrix PageRank
            pr = MatrixPageRank(device=device)
            pr.preprocess(adj_matrix)
            
            # Time execution
            mean_time, std_time = time_function(pr.run, n, n_runs=5)
            matrix_times.append(mean_time)
            matrix_stds.append(std_time)
            
            print(f"Graph size: {n}, Device: {device}, Matrix PageRank time: {mean_time:.6f} ± {std_time:.6f} s")
        
        results["matrix_pr_times"].append(matrix_times)
        results["matrix_pr_std"].append(matrix_stds)
        
        # NetworkX PageRank (only on CPU)
        nx_pr = NetworkXPageRank()
        nx_pr.preprocess(adj_matrix)
        
        # Time execution
        mean_time, std_time = time_function(nx_pr.run, n_runs=5)
        results["networkx_pr_times"].append(mean_time)
        results["networkx_pr_std"].append(std_time)
        
        print(f"Graph size: {n}, NetworkX PageRank time: {mean_time:.6f} ± {std_time:.6f} s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_benchmark_results(results, f"{output_dir}/pagerank_strong_scaling.npy")
    
    # Plot results
    plot_scaling(
        num_nodes, 
        [results["networkx_pr_times"]] + [
            [results["matrix_pr_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        ["NetworkX (CPU)"] + [f"Matrix PageRank ({device})" for device in devices],
        "PageRank Strong Scaling",
        f"{output_dir}/pagerank_strong_scaling.png"
    )
    
    # Plot speedup
    plot_speedup(
        num_nodes,
        results["networkx_pr_times"],
        [
            [results["matrix_pr_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        [f"Matrix PageRank ({device})" for device in devices],
        "PageRank Speedup vs. NetworkX",
        f"{output_dir}/pagerank_speedup.png"
    )

def run_strong_scaling_cc(adj_matrices, num_nodes, devices, output_dir="results/strong_scaling"):
    """
    Run strong scaling benchmark for Connected Components.
    
    Args:
        adj_matrices (list): List of adjacency matrices
        num_nodes (list): List of number of nodes for each graph
        devices (list): List of devices to test on
        output_dir (str): Directory to save results
    """
    results = {
        "graph_sizes": num_nodes,
        "devices": devices,
        "matrix_cc_times": [],
        "matrix_cc_std": [],
        "networkx_cc_times": [],
        "networkx_cc_std": []
    }
    
    for i, adj_matrix in enumerate(adj_matrices):
        n = num_nodes[i]
        
        matrix_times = []
        matrix_stds = []
        
        for device in devices:
            # Matrix Connected Components
            cc = MatrixConnectedComponents(device=device)
            cc.preprocess(adj_matrix)
            
            # Time execution
            mean_time, std_time = time_function(cc.run, n, n_runs=5)
            matrix_times.append(mean_time)
            matrix_stds.append(std_time)
            
            print(f"Graph size: {n}, Device: {device}, Matrix CC time: {mean_time:.6f} ± {std_time:.6f} s")
        
        results["matrix_cc_times"].append(matrix_times)
        results["matrix_cc_std"].append(matrix_stds)
        
        # NetworkX Connected Components (only on CPU)
        nx_cc = NetworkXCC()
        nx_cc.preprocess(adj_matrix)
        
        # Time execution
        mean_time, std_time = time_function(nx_cc.run, n_runs=5)
        results["networkx_cc_times"].append(mean_time)
        results["networkx_cc_std"].append(std_time)
        
        print(f"Graph size: {n}, NetworkX CC time: {mean_time:.6f} ± {std_time:.6f} s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_benchmark_results(results, f"{output_dir}/cc_strong_scaling.npy")
    
    # Plot results
    plot_scaling(
        num_nodes, 
        [results["networkx_cc_times"]] + [
            [results["matrix_cc_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        ["NetworkX (CPU)"] + [f"Matrix CC ({device})" for device in devices],
        "Connected Components Strong Scaling",
        f"{output_dir}/cc_strong_scaling.png"
    )
    
    # Plot speedup
    plot_speedup(
        num_nodes,
        results["networkx_cc_times"],
        [
            [results["matrix_cc_times"][i][j] for i in range(len(num_nodes))]
            for j in range(len(devices))
        ],
        [f"Matrix CC ({device})" for device in devices],
        "Connected Components Speedup vs. NetworkX",
        f"{output_dir}/cc_speedup.png"
    )

def main():
    # Load datasets
    data_dir = Path("data/synthetic/generated")
    graph_files = [
        "er_small.npz",
        "ba_small.npz",
        "er_medium.npz",
        "ba_medium.npz",
        "rmat_large.npz"
    ]
    
    adj_matrices = []
    num_nodes = []
    source_nodes = []
    
    for file in graph_files:
        adj_matrix = load_sparse_matrix(data_dir / file)
        adj_matrices.append(adj_matrix)
        num_nodes.append(adj_matrix.shape[0])
        source_nodes.append(0)  # Use node 0 as the source for BFS
    
    # Define devices to test
    devices = ["cpu", "cuda"]
    
    # Run benchmarks
    run_strong_scaling_bfs(adj_matrices, source_nodes, num_nodes, devices)
    run_strong_scaling_pagerank(adj_matrices, num_nodes, devices)
    run_strong_scaling_cc(adj_matrices, num_nodes, devices)

if __name__ == "__main__":
    main()