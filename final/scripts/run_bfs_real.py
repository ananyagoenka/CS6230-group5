#!/usr/bin/env python3
"""
Script to run BFS benchmarks on real-world graphs comparing traditional CPU,
sparse GPU, and CUDA-optimized sparse GPU implementations
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import gc
import psutil
import networkx as nx
import pandas as pd
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
from src.utils.graph_utils import (
    graph_to_adj_list,
    graph_to_sparse_adj_matrix_torch
)

def print_simplified_graph_stats(G):
    """Print only essential graph stats"""
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    if nx.is_directed(G):
        print("Graph is directed")
    else:
        print("Graph is undirected")

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

def load_graph_from_file(file_path, directed=False, node_id_type=int):
    """
    Load a graph from a file.
    
    Parameters:
    - file_path: Path to the graph file
    - directed: Whether the graph is directed or undirected
    - node_id_type: Type of node IDs (int, str, etc.)
    
    Returns:
    - NetworkX graph
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Create empty graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    print(f"Loading graph from {file_path}...")
    
    # Handle different file formats
    if file_ext in ['.txt', '.edges', '.csv']:
        # Try to load as edge list (most common format for SNAP datasets)
        try:
            # First check if file has headers
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                # Skip comment lines at the beginning
                while first_line.startswith('#'):
                    first_line = f.readline().strip()
                
                # Try to parse first line as edge
                parts = first_line.split()
                if len(parts) >= 2:
                    try:
                        # Check if the values can be converted to integers
                        node_id_type(parts[0])
                        node_id_type(parts[1])
                        has_header = False
                    except ValueError:
                        has_header = True
                else:
                    has_header = True
            
            # Load the edge list
            if file_ext == '.csv':
                # Assume CSV format with comma separator
                df = pd.read_csv(file_path, comment='#', header=0 if has_header else None)
                # Get column names or indices for the first two columns
                src_col = df.columns[0]
                dst_col = df.columns[1]
                # Add edges to the graph
                for _, row in df.iterrows():
                    G.add_edge(node_id_type(row[src_col]), node_id_type(row[dst_col]))
            else:
                # Assume space or tab separator
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                G.add_edge(node_id_type(parts[0]), node_id_type(parts[1]))
        except Exception as e:
            print(f"Error loading edge list: {e}")
            return None
    elif file_ext == '.gml':
        # Load GML format
        try:
            G = nx.read_gml(file_path)
        except Exception as e:
            print(f"Error loading GML file: {e}")
            return None
    elif file_ext == '.graphml':
        # Load GraphML format
        try:
            G = nx.read_graphml(file_path)
        except Exception as e:
            print(f"Error loading GraphML file: {e}")
            return None
    elif file_ext == '.gexf':
        # Load GEXF format
        try:
            G = nx.read_gexf(file_path)
        except Exception as e:
            print(f"Error loading GEXF file: {e}")
            return None
    else:
        print(f"Unsupported file format: {file_ext}")
        return None
    
    # Ensure node IDs are of the correct type
    if node_id_type != type(next(iter(G.nodes()), None)):
        G = nx.convert_node_labels_to_integers(G)
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def main():
    parser = argparse.ArgumentParser(description='Run BFS benchmarks on real-world graphs with CUDA-optimized implementation')
    parser.add_argument('--graph-files', type=str, nargs='+', required=True,
                        help='Paths to graph files to benchmark')
    parser.add_argument('--graphs-dir', type=str, default=None,
                        help='Directory containing graph files (will be combined with graph-files)')
    parser.add_argument('--directed', action='store_true',
                        help='Treat the graphs as directed (default: undirected)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for each benchmark')
    parser.add_argument('--verify', action='store_true', help='Verify implementation correctness before benchmarking')
    parser.add_argument('--skip-traditional', action='store_true', help='Skip traditional CPU implementation')
    parser.add_argument('--max-traditional-size', type=int, default=3000000,
                        help='Maximum graph size for traditional implementation (skipped if larger)')
    parser.add_argument('--only-gpu', action='store_true', help='Only run GPU implementations')
    parser.add_argument('--only-optimized', action='store_true', help='Only run optimized GPU implementation')
    parser.add_argument('--start-node-strategy', type=str, default='max-degree',
                        choices=['max-degree', 'random', 'first', 'central'],
                        help='Strategy to select start node for BFS')
    parser.add_argument('--output-file', type=str, default=None,
                        help='File to save benchmark results (CSV format)')
    parser.add_argument('--force-relabel', action='store_true',
                        help='Force relabeling of graph nodes to contiguous integers')
    
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
    
    # Get all graph files
    graph_files = args.graph_files
    if args.graphs_dir:
        # Add files from the directory
        for file in os.listdir(args.graphs_dir):
            file_path = os.path.join(args.graphs_dir, file)
            if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in [
                '.txt', '.edges', '.csv', '.gml', '.graphml', '.gexf'
            ]:
                graph_files.append(file_path)
    
    # Store results for summary
    all_results = {}
    
    # Process each graph file
    for graph_file in graph_files:
        graph_name = os.path.basename(graph_file)
        print(f"\n{'-'*60}")
        print(f"Benchmarking graph: {graph_name}")
        print(f"{'-'*60}")
        
        try:
            # Load the graph
            G = load_graph_from_file(graph_file, directed=args.directed)
            if G is None:
                print(f"Skipping graph {graph_name} due to loading error")
                continue
            
            # Always relabel nodes to ensure contiguous node IDs if forced or node IDs are not contiguous
            max_node_id = max(G.nodes())
            if args.force_relabel or max_node_id >= G.number_of_nodes():
                print(f"Node IDs are not contiguous (max ID: {max_node_id}, nodes count: {G.number_of_nodes()})")
                print("Relabeling graph to have contiguous node IDs...")
                G = nx.convert_node_labels_to_integers(G, first_label=0)
                print(f"Graph relabeled with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            print_simplified_graph_stats(G)
            
            # Choose a start node based on the strategy
            if args.start_node_strategy == 'max-degree':
                # Find node with max degree
                max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
                print(f"Start node: {max_degree_node} with degree {G.degree(max_degree_node)}")
                
                # IMPORTANT: Make sure to use a start node that's within the valid range
                # This should no longer be necessary after relabeling, but we'll keep it as a safety check
                if max_degree_node >= G.number_of_nodes():
                    print(f"Warning: Original max degree node ID {max_degree_node} is out of bounds.")
                    # Relabel the graph to have contiguous node IDs (0 to n-1)
                    print("Relabeling graph to have contiguous node IDs...")
                    mapping = {old_label: i for i, old_label in enumerate(G.nodes())}
                    G = nx.relabel_nodes(G, mapping)
                    # Find the new max degree node after relabeling
                    max_degree_node = max(G.degree(), key=lambda x: x[1])[0]
                    print(f"New start node after relabeling: {max_degree_node} with degree {G.degree(max_degree_node)}")
                
                start_node = max_degree_node
                
            elif args.start_node_strategy == 'random':
                start_node = np.random.choice(list(G.nodes()))
                print(f"Start node: {start_node} (randomly selected) with degree {G.degree(start_node)}")
            elif args.start_node_strategy == 'first':
                start_node = list(G.nodes())[0]
                print(f"Start node: {start_node} (first node) with degree {G.degree(start_node)}")
            elif args.start_node_strategy == 'central':
                # Choose a node with relatively high centrality (approximation for large graphs)
                # For large graphs, this is expensive, so just use degree as a proxy
                if G.number_of_nodes() > 10000:
                    # Use degree as a proxy for centrality
                    start_node = max(G.degree(), key=lambda x: x[1])[0]
                else:
                    # Calculate closeness centrality for a sample of nodes
                    sample_nodes = np.random.choice(list(G.nodes()), min(1000, G.number_of_nodes()), replace=False)
                    centrality = nx.closeness_centrality(G, nbunch=sample_nodes)
                    start_node = max(centrality.items(), key=lambda x: x[1])[0]
                print(f"Start node: {start_node} (central node) with degree {G.degree(start_node)}")
            
            # Print current memory usage
            print("\nMemory usage before running algorithms:")
            print_memory_usage()
            
            # Store results for this graph
            graph_results = {}
            
            # Skip traditional BFS for very large graphs or if requested
            run_traditional = (
                not args.skip_traditional and 
                not args.only_gpu and 
                not args.only_optimized and
                G.number_of_nodes() <= args.max_traditional_size
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
                graph_results['Traditional_BFS'] = trad_result
                print(f"Average time: {trad_result['avg_time']:.6f} seconds")
                print(f"Std dev: {trad_result['std_time']:.6f} seconds")
                print(f"Min time: {trad_result['min_time']:.6f} seconds")
                print(f"Max time: {trad_result['max_time']:.6f} seconds")
                baseline_time = trad_result['avg_time']
                
                # Free memory
                del adj_list
                gc.collect()
            else:
                if G.number_of_nodes() > args.max_traditional_size:
                    print(f"\nSkipping traditional BFS for large graph (nodes > {args.max_traditional_size})")
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
                try:
                    la_sparse_result, _ = run_test(
                        BFS.la_bfs_sparse_gpu,
                        adj_matrix_sparse,
                        start_node,
                        n_runs=args.runs
                    )
                    graph_results['LA_BFS_GPU_Sparse'] = la_sparse_result
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
                except Exception as e:
                    print(f"Error in standard sparse GPU BFS: {e}")
                    sparse_time = None
            else:
                print("\nSkipping standard sparse GPU BFS as requested")
                sparse_time = None
            
            # Run CUDA-optimized sparse GPU BFS
            print("\nRunning CUDA-optimized sparse BFS on GPU...")
            try:
                la_opt_sparse_result, _ = run_test(
                    BFS.la_bfs_sparse_gpu_optimized_v2_turbo,
                    adj_matrix_sparse,
                    start_node,
                    n_runs=args.runs
                )
                graph_results['LA_BFS_GPU_Sparse_Optimized'] = la_opt_sparse_result
                print(f"Average time: {la_opt_sparse_result['avg_time']:.6f} seconds")
                print(f"Std dev: {la_opt_sparse_result['std_time']:.6f} seconds")
                print(f"Min time: {la_opt_sparse_result['min_time']:.6f} seconds")
                print(f"Max time: {la_opt_sparse_result['max_time']:.6f} seconds")
                
                if baseline_time:
                    print(f"Speedup vs traditional: {baseline_time / la_opt_sparse_result['avg_time']:.2f}x")
                
                if sparse_time:
                    print(f"Speedup vs standard sparse: {sparse_time / la_opt_sparse_result['avg_time']:.2f}x")
            except Exception as e:
                print(f"Error in CUDA-optimized sparse BFS: {e}")
            
            # Store results for this graph
            all_results[graph_name] = graph_results
            
            # Clean up to free memory for next iteration
            del adj_matrix_sparse
            gc.collect()
            torch.cuda.empty_cache()
            
            # Print memory usage after tests
            print("\nMemory usage after completing tests:")
            print_memory_usage()
            
        except Exception as e:
            print(f"\nError processing graph {graph_name}: {e}")
            print("Skipping to next graph...")
            continue
    
    # Print summary of all results
    print("\n" + "="*80)
    print("SUMMARY FOR REAL-WORLD GRAPHS")
    print("="*80)
    
    # Print header
    headers = ["Graph", "Algorithm", "Avg Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)", "Speedup vs Trad", "Speedup vs Sparse"]
    print(f"{headers[0]:<30} {headers[1]:<30} {headers[2]:<15} {headers[3]:<10} {headers[4]:<15} {headers[5]:<15} {headers[6]:<15} {headers[7]:<15}")
    print("-" * 145)
    
    # Prepare data for CSV export if requested
    csv_data = []
    
    # Print results for each graph
    for graph_name in sorted(all_results.keys()):
        graph_results = all_results[graph_name]
        
        # Get baseline times if available
        trad_time = graph_results.get('Traditional_BFS', {}).get('avg_time', None)
        sparse_time = graph_results.get('LA_BFS_GPU_Sparse', {}).get('avg_time', None)
        
        # If traditional not available, use sparse as baseline
        if trad_time is None and sparse_time is not None:
            trad_time = sparse_time
        
        # Print results for each algorithm
        algorithms = ['Traditional_BFS', 'LA_BFS_GPU_Sparse', 'LA_BFS_GPU_Sparse_Optimized']
        for i, alg_name in enumerate(algorithms):
            if alg_name not in graph_results:
                continue
                
            result = graph_results[alg_name]
            
            # Calculate speedups
            if trad_time and alg_name != 'Traditional_BFS':
                speedup_vs_trad = trad_time / result['avg_time']
            else:
                speedup_vs_trad = 1.0 if alg_name == 'Traditional_BFS' else float('nan')
            
            if sparse_time and alg_name != 'LA_BFS_GPU_Sparse':
                speedup_vs_sparse = sparse_time / result['avg_time']
            else:
                speedup_vs_sparse = 1.0 if alg_name == 'LA_BFS_GPU_Sparse' else float('nan')
            
            # Print graph name only for the first algorithm
            graph_str = graph_name if i == 0 else ''
            
            # Format speedup values, handling NaN
            speedup_trad_str = f"{speedup_vs_trad:.2f}x" if not np.isnan(speedup_vs_trad) else "N/A"
            speedup_sparse_str = f"{speedup_vs_sparse:.2f}x" if not np.isnan(speedup_vs_sparse) else "N/A"
            
            print(f"{graph_str:<30} {alg_name:<30} {result['avg_time']:<15.6f} {result['std_time']:<10.6f} "
                  f"{result['min_time']:<15.6f} {result['max_time']:<15.6f} {speedup_trad_str:<15} {speedup_sparse_str:<15}")
            
            # Add data for CSV export
            if args.output_file:
                csv_data.append({
                    'Graph': graph_name,
                    'Algorithm': alg_name,
                    'Avg_Time_s': result['avg_time'],
                    'Std_Dev': result['std_time'],
                    'Min_Time_s': result['min_time'],
                    'Max_Time_s': result['max_time'],
                    'Speedup_vs_Trad': speedup_vs_trad if not np.isnan(speedup_vs_trad) else None,
                    'Speedup_vs_Sparse': speedup_vs_sparse if not np.isnan(speedup_vs_sparse) else None
                })
        
        print("-" * 145)
    
    # Export results to CSV if requested
    if args.output_file and csv_data:
        try:
            df = pd.DataFrame(csv_data)
            df.to_csv(args.output_file, index=False)
            print(f"\nResults exported to {args.output_file}")
        except Exception as e:
            print(f"\nError exporting results to CSV: {e}")
    
    print("\nBenchmark complete!")

if __name__ == '__main__':
    main()