import numpy as np
import torch

def verify_bfs_correctness(results):
    # Use traditional BFS as reference
    ref_impl = 'Traditional_BFS'
    if ref_impl not in results:
        print("Error: Reference implementation not found")
        return False
    
    ref_visited, ref_distances = results[ref_impl]
    
    # For each implementation, create a list of (node, distance) pairs
    # and sort first by distance, then by node ID
    ref_pairs = [(int(node), ref_distances[node]) for node in ref_visited]
    ref_pairs.sort(key=lambda x: (x[1], x[0]))  # Sort by distance, then by node ID
    
    # Compare other implementations against reference
    for impl, (visited, distances) in results.items():
        if impl == ref_impl:
            continue
        
        # Convert to pairs and sort
        impl_pairs = [(int(node), distances[node]) for node in visited]
        impl_pairs.sort(key=lambda x: (x[1], x[0]))  # Sort by distance, then by node ID
        
        # Compare sorted pairs
        if impl_pairs != ref_pairs:
            print(f"Error: {impl} node-distance pairs differ from {ref_impl}")
            
            # Find first difference
            for i in range(min(len(ref_pairs), len(impl_pairs))):
                if ref_pairs[i] != impl_pairs[i]:
                    print(f"First difference at index {i}: {ref_pairs[i]} vs {impl_pairs[i]}")
                    break
            
            return False
        
        print(f"{impl} node-distance pairs match reference after sorting")
    
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

import numpy as np
import torch

def verify_connected_components_correctness(results):
    """
    Verify that different Connected Components implementations return equivalent results.
    Two implementations are considered equivalent if they divide the graph into the same connected components,
    even if the component IDs are different.
    
    Parameters:
    -----------
    results : dict
        Dictionary of CC results from different implementations, where each result is a tuple
        (components, component_map)
        
    Returns:
    --------
    is_correct : bool
        Whether all implementations return equivalent results
    """
    # Use traditional CC as reference
    ref_impl = 'Traditional_CC'
    if ref_impl not in results:
        print("Error: Reference implementation not found")
        return False
    
    ref_components, ref_component_map = results[ref_impl]
    
    # Create a reference relation: nodes in the same component
    ref_same_component = {}
    for node1 in ref_component_map:
        for node2 in ref_component_map:
            if node1 <= node2:  # Avoid duplicates (i,j) and (j,i)
                key = (node1, node2)
                ref_same_component[key] = (ref_component_map[node1] == ref_component_map[node2])
    
    # Compare other implementations against reference
    for impl, (components, component_map) in results.items():
        if impl == ref_impl:
            continue
        
        # Check if connectivity relation is the same
        same_component = {}
        for node1 in component_map:
            for node2 in component_map:
                if node1 <= node2:  # Avoid duplicates
                    key = (node1, node2)
                    same_component[key] = (component_map[node1] == component_map[node2])
        
        # Compare relations
        is_equivalent = True
        mismatches = []
        for key, value in ref_same_component.items():
            if key not in same_component:
                print(f"Error: Node pair {key} missing in {impl}")
                is_equivalent = False
                break
            
            if same_component[key] != value:
                mismatches.append(key)
                is_equivalent = False
        
        if not is_equivalent:
            print(f"Error: {impl} is not equivalent to reference {ref_impl}")
            if mismatches:
                print(f"First few mismatches: {mismatches[:5]}")
            return False
        
        # Check number of components
        if len(ref_components) != len(components):
            print(f"Warning: {impl} has different number of components than {ref_impl}")
            print(f"  {ref_impl}: {len(ref_components)}, {impl}: {len(components)}")
            # This can be valid if the component IDs are not consecutive integers
            # So we don't fail just for this
        
        print(f"{impl} is equivalent to reference {ref_impl}")
    
    return True

def print_cc_stats(results):
    """
    Print statistics about Connected Components results
    
    Parameters:
    -----------
    results : dict
        Dictionary of algorithm results from different implementations
    """
    for impl, result in results.items():
        print(f"\n{impl} statistics:")
        
        components, component_map = result
        
        # Count components
        num_components = len(components)
        print(f"  Number of components: {num_components}")
        
        # Get component sizes
        component_sizes = [len(component) for component in components]
        
        # Print component size stats
        print(f"  Largest component size: {max(component_sizes) if component_sizes else 0}")
        print(f"  Smallest component size: {min(component_sizes) if component_sizes else 0}")
        print(f"  Average component size: {sum(component_sizes)/len(component_sizes) if component_sizes else 0:.2f}")
        
        # Print distribution of component sizes
        size_counts = {}
        for size in component_sizes:
            if size not in size_counts:
                size_counts[size] = 0
            size_counts[size] += 1
        
        # Print most common component sizes
        if size_counts:
            sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most common component sizes:")
            for size, count in sorted_sizes[:5]:  # Top 5 most common sizes
                print(f"    Size {size}: {count} components")
                
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