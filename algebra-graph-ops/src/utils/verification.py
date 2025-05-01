import numpy as np
import torch

def verify_bfs_correctness(results):
    """
    Verify that different BFS implementations return the same results
    
    Parameters:
    -----------
    results : dict
        Dictionary of BFS results from different implementations
        
    Returns:
    --------
    is_correct : bool
        Whether all implementations return the same results
    """
    # Use traditional BFS as reference
    ref_impl = 'Traditional_BFS'
    if ref_impl not in results:
        print("Error: Reference implementation not found")
        return False
    
    ref_visited, ref_distances = results[ref_impl]
    
    # Compare other implementations against reference
    for impl, (visited, distances) in results.items():
        if impl == ref_impl:
            continue
            
        # Check if visited nodes are the same (order matters for BFS)
        if visited != ref_visited:
            print(f"Error: {impl} visited nodes differ from {ref_impl}")
            print(f"First 10 nodes in reference: {ref_visited[:10]}")
            print(f"First 10 nodes in {impl}: {visited[:10]}")
            if len(ref_visited) != len(visited):
                print(f"Length mismatch: reference = {len(ref_visited)}, {impl} = {len(visited)}")
            return False
            
        # Check if distances are the same
        for node, dist in ref_distances.items():
            if node not in distances:
                print(f"Error: Node {node} missing in {impl} distances")
                return False
            if distances[node] != dist:
                print(f"Error: {impl} distance for node {node} is {distances[node]}, but reference is {dist}")
                return False
    
    return True

def verify_pagerank_correctness(results, tolerance=1e-5):
    """
    Verify that different PageRank implementations return similar results
    
    Parameters:
    -----------
    results : dict
        Dictionary of PageRank results from different implementations
    tolerance : float
        Maximum allowed difference between rank values
        
    Returns:
    --------
    is_correct : bool
        Whether all implementations return similar results
    """
    # Use traditional PageRank as reference
    ref_impl = 'Traditional_PageRank'
    if ref_impl not in results:
        print("Error: Reference implementation not found")
        return False
    
    ref_ranks, _ = results[ref_impl]
    
    # Compare other implementations against reference
    for impl, (ranks, _) in results.items():
        if impl == ref_impl:
            continue
            
        # For traditional implementation with dict output, convert to array
        if isinstance(ref_ranks, dict) and isinstance(ranks, (np.ndarray, torch.Tensor)):
            ref_array = np.zeros(len(ref_ranks))
            for node, rank in ref_ranks.items():
                ref_array[node] = rank
            ref_ranks_compare = ref_array
        else:
            ref_ranks_compare = ref_ranks
            
        # Convert to numpy for comparison
        if isinstance(ranks, torch.Tensor):
            ranks_compare = ranks.cpu().numpy()
        else:
            ranks_compare = ranks
            
        # Check if ranks are similar (within tolerance)
        if isinstance(ranks_compare, dict):
            # Compare dictionaries
            max_diff = 0
            max_diff_node = None
            for node, rank in ref_ranks.items():
                if node not in ranks_compare:
                    print(f"Error: Node {node} missing in {impl} ranks")
                    return False
                
                diff = abs(ranks_compare[node] - rank)
                if diff > max_diff:
                    max_diff = diff
                    max_diff_node = node
                    
                if diff > tolerance:
                    print(f"Error: {impl} rank for node {node} is {ranks_compare[node]}, but reference is {rank}")
                    print(f"Difference: {diff}, which exceeds tolerance of {tolerance}")
                    return False
            
            print(f"Maximum difference for {impl}: {max_diff} at node {max_diff_node}")
        else:
            # Compare arrays
            if not np.allclose(ranks_compare, ref_ranks_compare, atol=tolerance):
                # Find the maximum difference
                if len(ranks_compare) == len(ref_ranks_compare):
                    diff = np.abs(ranks_compare - ref_ranks_compare)
                    max_diff_idx = np.argmax(diff)
                    max_diff = diff[max_diff_idx]
                    print(f"Error: {impl} ranks differ from {ref_impl}")
                    print(f"Maximum difference: {max_diff} at index {max_diff_idx}")
                    print(f"{impl} value: {ranks_compare[max_diff_idx]}, reference value: {ref_ranks_compare[max_diff_idx]}")
                else:
                    print(f"Error: {impl} ranks have different length than {ref_impl}")
                return False
    
    return True

def print_algorithm_stats(results):
    """
    Print statistics about algorithm results
    
    Parameters:
    -----------
    results : dict
        Dictionary of algorithm results from different implementations
    """
    for impl, result in results.items():
        print(f"\n{impl} statistics:")
        
        # For BFS
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            visited, distances = result
            print(f"  Visited nodes: {len(visited)}")
            print(f"  Max distance: {max(distances.values())}")
            levels = {}
            for node, distance in distances.items():
                if distance not in levels:
                    levels[distance] = 0
                levels[distance] += 1
            print(f"  Nodes per level: {levels}")
        
        # For PageRank
        elif isinstance(result, tuple) and len(result) == 2 and (isinstance(result[0], dict) or 
                                                            isinstance(result[0], np.ndarray) or 
                                                            isinstance(result[0], torch.Tensor)):
            ranks, iterations = result
            
            if isinstance(ranks, dict):
                rank_values = list(ranks.values())
            elif isinstance(ranks, torch.Tensor):
                rank_values = ranks.cpu().numpy()
            else:
                rank_values = ranks
                
            if isinstance(rank_values, list):
                rank_values = np.array(rank_values)
                
            print(f"  Iterations: {iterations}")
            print(f"  Min rank: {np.min(rank_values)}")
            print(f"  Max rank: {np.max(rank_values)}")
            print(f"  Mean rank: {np.mean(rank_values)}")
            print(f"  Std dev: {np.std(rank_values)}")