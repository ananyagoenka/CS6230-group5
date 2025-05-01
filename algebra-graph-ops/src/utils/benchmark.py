import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import json
import os
from datetime import datetime

class Benchmark:
    """
    Class for running benchmarks and analyzing results
    """
    
    def __init__(self, name, save_dir="results"):
        """
        Initialize benchmark
        
        Parameters:
        -----------
        name : str
            Name of the benchmark
        save_dir : str
            Directory to save results (default: "results")
        """
        self.name = name
        self.save_dir = save_dir
        self.results = {}
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def run_test(self, func, *args, n_runs=5, **kwargs):
        """
        Run a benchmark test
        
        Parameters:
        -----------
        func : callable
            Function to benchmark
        *args : tuple
            Positional arguments to pass to the function
        n_runs : int
            Number of runs for averaging (default: 5)
        **kwargs : dict
            Keyword arguments to pass to the function
            
        Returns:
        --------
        result : dict
            Dictionary with benchmark results
        """
        times = []
        results = []
        
        # Warm-up run (not timed)
        _ = func(*args, **kwargs)
        
        # Timed runs
        for _ in range(n_runs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            times.append(end_time - start_time)
            results.append(result)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Return result dictionary
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "times": times,
            "result": results[0]  # Return first result
        }
    
    def add_result(self, algorithm, graph_name, graph_size, result):
        """
        Add a benchmark result
        
        Parameters:
        -----------
        algorithm : str
            Name of the algorithm
        graph_name : str
            Name of the graph
        graph_size : int
            Size of the graph (number of nodes)
        result : dict
            Benchmark result from run_test
        """
        if algorithm not in self.results:
            self.results[algorithm] = {}
        
        if graph_name not in self.results[algorithm]:
            self.results[algorithm][graph_name] = {}
        
        self.results[algorithm][graph_name][graph_size] = result
    
    def save_results(self, filename=None):
        """
        Save benchmark results to a JSON file
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save results (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
        
        # Convert numpy values to Python native types
        results_copy = {}
        for alg, graphs in self.results.items():
            results_copy[alg] = {}
            for graph, sizes in graphs.items():
                results_copy[alg][graph] = {}
                for size, result in sizes.items():
                    results_copy[alg][graph][size] = {
                        "avg_time": float(result["avg_time"]),
                        "std_time": float(result["std_time"]),
                        "min_time": float(result["min_time"]),
                        "max_time": float(result["max_time"]),
                        "times": [float(t) for t in result["times"]]
                    }
        
        # Save to file
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {os.path.join(self.save_dir, filename)}")
    
    def load_results(self, filename):
        """
        Load benchmark results from a JSON file
        
        Parameters:
        -----------
        filename : str
            Name of the file to load results
        """
        with open(os.path.join(self.save_dir, filename), 'r') as f:
            self.results = json.load(f)
        
        print(f"Results loaded from {os.path.join(self.save_dir, filename)}")
    
    def print_results(self, algorithms=None, graph_names=None, graph_sizes=None):
        """
        Print benchmark results in a tabular format
        
        Parameters:
        -----------
        algorithms : list, optional
            List of algorithms to include (default: all)
        graph_names : list, optional
            List of graph names to include (default: all)
        graph_sizes : list, optional
            List of graph sizes to include (default: all)
        """
        # Filter algorithms
        if algorithms is None:
            algorithms = list(self.results.keys())
        
        # Prepare data for tabulation
        table_data = []
        
        for alg in algorithms:
            if alg not in self.results:
                continue
            
            # Filter graph names
            if graph_names is None:
                graph_names_filtered = list(self.results[alg].keys())
            else:
                graph_names_filtered = [g for g in graph_names if g in self.results[alg]]
            
            for graph in graph_names_filtered:
                # Filter graph sizes
                if graph_sizes is None:
                    sizes = list(self.results[alg][graph].keys())
                else:
                    sizes = [s for s in graph_sizes if str(s) in self.results[alg][graph]]
                
                for size in sizes:
                    result = self.results[alg][graph][str(size)]
                    table_data.append([
                        alg,
                        graph,
                        size,
                        f"{result['avg_time']:.6f}",
                        f"{result['std_time']:.6f}",
                        f"{result['min_time']:.6f}",
                        f"{result['max_time']:.6f}"
                    ])
        
        # Print table
        headers = ["Algorithm", "Graph", "Size", "Avg Time (s)", "Std Dev (s)", "Min Time (s)", "Max Time (s)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def plot_comparison(self, graph_name, algorithms=None, graph_sizes=None, log_scale=True, 
                        save_file=None, show_plot=True):
        """
        Plot performance comparison of different algorithms
        
        Parameters:
        -----------
        graph_name : str
            Name of the graph to compare
        algorithms : list, optional
            List of algorithms to include (default: all)
        graph_sizes : list, optional
            List of graph sizes to include (default: all)
        log_scale : bool
            Whether to use logarithmic scale for y-axis (default: True)
        save_file : str, optional
            Path to save the plot (default: None)
        show_plot : bool
            Whether to display the plot (default: True)
        """
        # Set plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # Filter algorithms
        if algorithms is None:
            algorithms = list(self.results.keys())
        
        # Prepare data for plotting
        plot_data = []
        
        for alg in algorithms:
            if alg not in self.results or graph_name not in self.results[alg]:
                continue
            
            # Filter graph sizes
            if graph_sizes is None:
                sizes = sorted([int(s) for s in self.results[alg][graph_name].keys()])
            else:
                sizes = sorted([s for s in graph_sizes if str(s) in self.results[alg][graph_name]])
            
            for size in sizes:
                result = self.results[alg][graph_name][str(size)]
                plot_data.append({
                    "Algorithm": alg,
                    "Graph Size": size,
                    "Execution Time (s)": result["avg_time"],
                    "Std Dev": result["std_time"]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create plot
        ax = sns.lineplot(x="Graph Size", y="Execution Time (s)", hue="Algorithm", 
                        data=df, marker="o", markersize=8)
        
        # Add error bars
        for alg in df["Algorithm"].unique():
            alg_data = df[df["Algorithm"] == alg]
            plt.errorbar(alg_data["Graph Size"], alg_data["Execution Time (s)"],
                        yerr=alg_data["Std Dev"], fmt="none", alpha=0.3)
        
        # Set axis labels and title
        plt.xlabel("Number of Nodes", fontsize=14)
        plt.ylabel("Execution Time (seconds)", fontsize=14)
        plt.title(f"Performance Comparison - {graph_name}", fontsize=16)
        
        # Set log scale if requested
        if log_scale:
            plt.yscale("log")
        
        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Improve legend
        plt.legend(title="Algorithm", fontsize=12)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save_file is not None:
            plt.savefig(os.path.join(self.save_dir, save_file), dpi=300, bbox_inches="tight")
            print(f"Plot saved to {os.path.join(self.save_dir, save_file)}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_speedup(self, reference_algorithm, graph_name, algorithms=None, graph_sizes=None,
                    save_file=None, show_plot=True):
        """
        Plot speedup relative to a reference algorithm
        
        Parameters:
        -----------
        reference_algorithm : str
            Name of the reference algorithm to compare against
        graph_name : str
            Name of the graph to compare
        algorithms : list, optional
            List of algorithms to include (default: all)
        graph_sizes : list, optional
            List of graph sizes to include (default: all)
        save_file : str, optional
            Path to save the plot (default: None)
        show_plot : bool
            Whether to display the plot (default: True)
        """
        # Set plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # Filter algorithms
        if algorithms is None:
            algorithms = list(self.results.keys())
        
        if reference_algorithm not in algorithms:
            print(f"Reference algorithm '{reference_algorithm}' not found in results")
            return
        
        # Remove reference algorithm from the list to avoid comparing it with itself
        algorithms = [alg for alg in algorithms if alg != reference_algorithm]
        
        # Prepare data for plotting
        plot_data = []
        
        for alg in algorithms:
            if (alg not in self.results or graph_name not in self.results[alg] or
                reference_algorithm not in self.results or graph_name not in self.results[reference_algorithm]):
                continue
            
            # Filter graph sizes
            if graph_sizes is None:
                # Find common sizes between reference and current algorithm
                ref_sizes = set(self.results[reference_algorithm][graph_name].keys())
                alg_sizes = set(self.results[alg][graph_name].keys())
                sizes = sorted([int(s) for s in ref_sizes.intersection(alg_sizes)])
            else:
                sizes = sorted([s for s in graph_sizes 
                              if str(s) in self.results[alg][graph_name] 
                              and str(s) in self.results[reference_algorithm][graph_name]])
            
            for size in sizes:
                ref_result = self.results[reference_algorithm][graph_name][str(size)]
                alg_result = self.results[alg][graph_name][str(size)]
                
                # Calculate speedup
                speedup = ref_result["avg_time"] / alg_result["avg_time"]
                
                plot_data.append({
                    "Algorithm": alg,
                    "Graph Size": size,
                    "Speedup": speedup
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(plot_data)
        
        # Create plot
        ax = sns.lineplot(x="Graph Size", y="Speedup", hue="Algorithm", 
                        data=df, marker="o", markersize=8)
        
        # Set axis labels and title
        plt.xlabel("Number of Nodes", fontsize=14)
        plt.ylabel(f"Speedup vs. {reference_algorithm}", fontsize=14)
        plt.title(f"Speedup Comparison - {graph_name}", fontsize=16)
        
        # Add horizontal line at y=1 (no speedup)
        plt.axhline(y=1, color="red", linestyle="--", alpha=0.7)
        
        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Improve legend
        plt.legend(title="Algorithm", fontsize=12)
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot if requested
        if save_file is not None:
            plt.savefig(os.path.join(self.save_dir, save_file), dpi=300, bbox_inches="tight")
            print(f"Plot saved to {os.path.join(self.save_dir, save_file)}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()