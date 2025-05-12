"""
Utilities for verifying the correctness of graph algorithm implementations
"""

import numpy as np
from collections import defaultdict

def verify_bfs_correctness(bfs_results):
    """
    Verify that all BFS implementations produce the same results
    
    Parameters:
    -----------
    bfs_results : dict
        Dictionary mapping implementation names to BFS results (visited, distances)
        
    Returns:
    --------
    is_correct : bool
        True if all implementations produce identical results
    """
    if not bfs_results:
        return True
    
    # Get reference implementation (first in the dictionary)
    reference_name = list(bfs_results.keys())[0]
    reference_visited, reference_distances = bfs_results[reference_name]
    
    for name, (visited, distances) in bfs_results.items():
        if name == reference_name:
            continue
        
        # Check if all nodes in the reference implementation have the same distance
        for node, distance in reference_distances.items():
            if node not in distances or distances[node] != distance:
                print(f"Mismatch in {name} for node {node}: Expected {distance}, Got {distances.get(node, 'missing')}")
                return False
        
        # Check if all nodes in this implementation are in the reference
        for node in distances:
            if node not in reference_distances:
                print(f"Node {node} in {name} but not in reference implementation")
                return False
    
    return True

def verify_pagerank_correctness(pagerank_results, tolerance=1e-5):
    """
    Verify that all PageRank implementations produce similar results
    
    Parameters:
    -----------
    pagerank_results : dict
        Dictionary mapping implementation names to PageRank results (ranks, iterations)
    tolerance : float
        Maximum allowed difference in PageRank values
        
    Returns:
    --------
    is_correct : bool
        True if all implementations produce similar results within tolerance
    """
    if not pagerank_results:
        return True
    
    # Get reference implementation (first in the dictionary)
    reference_name = list(pagerank_results.keys())[0]
    reference_ranks, _ = pagerank_results[reference_name]
    
    for name, (ranks, _) in pagerank_results.items():
        if name == reference_name:
            continue
        
        # Check if all nodes in the reference implementation have similar rank
        for node, rank in reference_ranks.items():
            if node not in ranks:
                print(f"Node {node} missing in {name}")
                return False
            
            # Convert to float if needed (to handle numpy/torch types)
            ref_rank = float(rank)
            test_rank = float(ranks[node])
            
            # Check if the ranks are similar within tolerance
            if abs(ref_rank - test_rank) > tolerance:
                print(f"Rank mismatch in {name} for node {node}: "
                      f"Expected {ref_rank}, Got {test_rank}, Diff {abs(ref_rank - test_rank)}")
                return False
        
        # Check if all nodes in this implementation are in the reference
        for node in ranks:
            if node not in reference_ranks:
                print(f"Node {node} in {name} but not in reference implementation")
                return False
    
    return True

def print_algorithm_stats(results):
    """
    Print detailed statistics for algorithm results
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping implementation names to algorithm results
    """
    print("\nAlgorithm Statistics:")
    print("=" * 80)
    
    if not results:
        print("No results to display.")
        return
    
    # BFS specific statistics
    if isinstance(list(results.values())[0][0], list):  # First element is a list of visited nodes
        max_distances = {}
        visited_counts = {}
        
        for name, (visited, distances) in results.items():
            max_dist = max(d for d in distances.values() if d != float('infinity'))
            max_distances[name] = max_dist
            visited_counts[name] = len(visited)
        
        print(f"{'Implementation':<30} {'Max Distance':<15} {'Visited Nodes':<15}")
        print("-" * 80)
        
        for name in results:
            print(f"{name:<30} {max_distances[name]:<15} {visited_counts[name]:<15}")
    
    # PageRank specific statistics
    elif isinstance(list(results.values())[0][0], dict):  # First element is a dictionary of ranks
        iterations = {}
        min_ranks = {}
        max_ranks = {}
        avg_ranks = {}
        
        for name, (ranks, iters) in results.items():
            iterations[name] = iters
            rank_values = [float(r) for r in ranks.values()]
            min_ranks[name] = min(rank_values)
            max_ranks[name] = max(rank_values)
            avg_ranks[name] = sum(rank_values) / len(rank_values)
        
        print(f"{'Implementation':<30} {'Iterations':<10} {'Min Rank':<15} {'Max Rank':<15} {'Avg Rank':<15}")
        print("-" * 85)
        
        for name in results:
            print(f"{name:<30} {iterations[name]:<10} {min_ranks[name]:<15.8f} {max_ranks[name]:<15.8f} {avg_ranks[name]:<15.8f}")
    
    else:
        print("Unknown result format. Cannot display statistics.")