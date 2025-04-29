#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <stdio.h>

// Error checking macro
#define CHECK_CUDA(func)                                               \
{                                                                      \
    cudaError_t status = (func);                                       \
    if (status != cudaSuccess) {                                       \
        printf("CUDA error at line %d: %s\n", __LINE__,                \
               cudaGetErrorString(status));                            \
        return EXIT_FAILURE;                                           \
    }                                                                  \
}

extern "C" {

// Specialized BFS kernel: Fused sparse matrix-vector multiplication and distance updating
__global__ void bfs_fused_kernel(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ current_frontier,
    int* __restrict__ next_frontier,
    int* __restrict__ distances,
    int* __restrict__ predecessors,
    const int level,
    const int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        // Skip if not in current frontier
        if (!current_frontier[tid]) return;
        
        int row_start = csr_row_ptr[tid];
        int row_end = csr_row_ptr[tid + 1];
        
        // Explore neighbors
        for (int j = row_start; j < row_end; j++) {
            int neighbor = csr_col_ind[j];
            
            // Check if neighbor hasn't been visited (distance is infinity/max_int)
            if (distances[neighbor] == INT_MAX) {
                // Update distance atomically (avoid race conditions when multiple threads update same node)
                if (atomicCAS(&distances[neighbor], INT_MAX, level + 1) == INT_MAX) {
                    // If we were the first to update the distance, also update predecessor
                    predecessors[neighbor] = tid;
                    // Add to next frontier
                    next_frontier[neighbor] = 1;
                }
            }
        }
    }
}

// Optimized BFS kernel for multi-GPU execution
__global__ void bfs_multi_gpu_kernel(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ node_partition,  // Node partition info for multi-GPU
    const int* __restrict__ current_frontier,
    int* __restrict__ next_frontier,
    int* __restrict__ distances,
    int* __restrict__ predecessors,
    const int level,
    const int num_nodes,
    const int gpu_id,
    const int num_gpus
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        // Check if this node belongs to the current GPU's partition
        if (node_partition[tid] != gpu_id) return;
        
        // Skip if not in current frontier
        if (!current_frontier[tid]) return;
        
        int row_start = csr_row_ptr[tid];
        int row_end = csr_row_ptr[tid + 1];
        
        // Explore neighbors
        for (int j = row_start; j < row_end; j++) {
            int neighbor = csr_col_ind[j];
            
            // Check if neighbor hasn't been visited
            if (distances[neighbor] == INT_MAX) {
                // Update distance atomically
                if (atomicCAS(&distances[neighbor], INT_MAX, level + 1) == INT_MAX) {
                    // If we were the first to update the distance, also update predecessor
                    predecessors[neighbor] = tid;
                    // Add to next frontier
                    next_frontier[neighbor] = 1;
                }
            }
        }
    }
}

// Wrapper function for BFS kernel
int bfs_kernel_launch(
    int* csr_row_ptr,         // CSR row pointers
    int* csr_col_ind,         // CSR column indices
    int* current_frontier,    // Current frontier
    int* next_frontier,       // Next frontier (output)
    int* distances,           // Distances array
    int* predecessors,        // Predecessors array
    int level,                // Current BFS level
    int num_nodes             // Number of nodes
) {
    // Configure kernel
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    
    // Launch kernel
    bfs_fused_kernel<<<numBlocks, blockSize>>>(
        csr_row_ptr, csr_col_ind, current_frontier,
        next_frontier, distances, predecessors,
        level, num_nodes
    );
    
    // Check for errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return 0;
}

// Wrapper function for multi-GPU BFS kernel
int bfs_multi_gpu_kernel_launch(
    int* csr_row_ptr,         // CSR row pointers
    int* csr_col_ind,         // CSR column indices
    int* node_partition,      // Node partition info
    int* current_frontier,    // Current frontier
    int* next_frontier,       // Next frontier (output)
    int* distances,           // Distances array
    int* predecessors,        // Predecessors array
    int level,                // Current BFS level
    int num_nodes,            // Number of nodes
    int gpu_id,               // Current GPU ID
    int num_gpus              // Total number of GPUs
) {
    // Configure kernel
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    
    // Launch kernel
    bfs_multi_gpu_kernel<<<numBlocks, blockSize>>>(
        csr_row_ptr, csr_col_ind, node_partition,
        current_frontier, next_frontier, distances, predecessors,
        level, num_nodes, gpu_id, num_gpus
    );
    
    // Check for errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return 0;
}

} // extern "C"