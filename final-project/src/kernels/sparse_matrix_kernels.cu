#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

// cuSPARSE error checking macro
#define CHECK_CUSPARSE(func)                                           \
{                                                                      \
    cusparseStatus_t status = (func);                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        printf("cuSPARSE error at line %d: %s\n", __LINE__,            \
               cusparseGetErrorString(status));                        \
        return EXIT_FAILURE;                                           \
    }                                                                  \
}

extern "C" {

// Optimized SpMV (Sparse Matrix-Vector Multiplication) for BFS
int optimized_spmv_for_bfs(
    int* csr_row_ptr,      // CSR row pointers
    int* csr_col_ind,      // CSR column indices
    float* csr_values,     // CSR values
    int num_rows,          // Number of rows
    int num_cols,          // Number of columns
    int nnz,               // Number of non-zeros
    float* x,              // Input vector
    float* y               // Output vector
) {
    // Initialize cuSPARSE
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    
    // Create description for matrices
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, num_rows, num_cols, nnz,
        csr_row_ptr, csr_col_ind, csr_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    ));
    
    // Create dense vectors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, y, CUDA_R_32F));
    
    // Allocate workspace buffer
    size_t bufferSize = 0;
    void* buffer = NULL;
    
    // Get required buffer size
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    ));
    
    // Allocate buffer
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
    
    // Execute SpMV
    CHECK_CUSPARSE(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        buffer
    ));
    
    // Clean up
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    
    if (buffer) CHECK_CUDA(cudaFree(buffer));
    
    return 0;
}

// Optimized SpMM (Sparse Matrix-Matrix Multiplication) for PageRank
int optimized_spmm(
    int* csr_row_ptr_A,    // CSR row pointers for matrix A
    int* csr_col_ind_A,    // CSR column indices for matrix A
    float* csr_values_A,   // CSR values for matrix A
    int num_rows_A,        // Number of rows in A
    int num_cols_A,        // Number of columns in A
    int nnz_A,             // Number of non-zeros in A
    float* B,              // Dense matrix B
    int num_cols_B,        // Number of columns in B
    float* C,              // Output dense matrix C
    int ldb,               // Leading dimension of B
    int ldc                // Leading dimension of C
) {
    // Initialize cuSPARSE
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    
    // Create description for matrices
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, num_rows_A, num_cols_A, nnz_A,
        csr_row_ptr_A, csr_col_ind_A, csr_values_A,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    ));
    
    // Create dense matrices B and C
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB, num_cols_A, num_cols_B, ldb, B,
        CUDA_R_32F, CUSPARSE_ORDER_COL
    ));
    
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC, num_rows_A, num_cols_B, ldc, C,
        CUDA_R_32F, CUSPARSE_ORDER_COL
    ));
    
    // Allocate workspace buffer
    size_t bufferSize = 0;
    void* buffer = NULL;
    
    // Get required buffer size
    float alpha = 1.0f;
    float beta = 0.0f;
    
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize
    ));
    
    // Allocate buffer
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));
    
    // Execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        buffer
    ));
    
    // Clean up
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    
    if (buffer) CHECK_CUDA(cudaFree(buffer));
    
    return 0;
}

// Specialized kernel for BFS frontier expansion
__global__ void bfs_frontier_kernel(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ current_frontier,
    const int* __restrict__ distances,
    int* __restrict__ next_frontier,
    int* __restrict__ new_distances,
    const int num_nodes,
    const int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes && current_frontier[tid]) {
        // For each node in the current frontier
        int row_start = csr_row_ptr[tid];
        int row_end = csr_row_ptr[tid + 1];
        
        // For each neighbor
        for (int j = row_start; j < row_end; j++) {
            int neighbor = csr_col_ind[j];
            
            // If neighbor hasn't been visited yet
            if (distances[neighbor] == INT_MAX) {
                // Mark for next frontier and update distance
                next_frontier[neighbor] = 1;
                atomicMin(&new_distances[neighbor], level + 1);
            }
        }
    }
}

// Wrapper function for BFS kernel
int bfs_frontier_expansion(
    int* csr_row_ptr,         // CSR row pointers
    int* csr_col_ind,         // CSR column indices
    int* current_frontier,    // Current frontier
    int* distances,           // Current distances
    int* next_frontier,       // Next frontier (output)
    int* new_distances,       // New distances (output)
    int num_nodes,            // Number of nodes
    int level                 // Current BFS level
) {
    // Configure kernel
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    
    // Launch kernel
    bfs_frontier_kernel<<<numBlocks, blockSize>>>(
        csr_row_ptr, csr_col_ind, current_frontier,
        distances, next_frontier, new_distances,
        num_nodes, level
    );
    
    // Check for errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return 0;
}

// Connected Components specific kernel for label propagation
__global__ void cc_label_propagation_kernel(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    int* __restrict__ labels,
    bool* __restrict__ changed,
    const int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes) {
        int current_label = labels[tid];
        int row_start = csr_row_ptr[tid];
        int row_end = csr_row_ptr[tid + 1];
        
        // For each neighbor
        for (int j = row_start; j < row_end; j++) {
            int neighbor = csr_col_ind[j];
            int neighbor_label = labels[neighbor];
            
            // If neighbor has smaller label, update current node's label
            if (neighbor_label < current_label) {
                labels[tid] = neighbor_label;
                *changed = true;
                break;
            }
        }
    }
}

// Wrapper function for Connected Components kernel
int cc_label_propagation(
    int* csr_row_ptr,      // CSR row pointers
    int* csr_col_ind,      // CSR column indices
    int* labels,           // Node labels
    bool* changed,         // Flag to track changes
    int num_nodes          // Number of nodes
) {
    // Configure kernel
    int blockSize = 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;
    
    // Launch kernel
    cc_label_propagation_kernel<<<numBlocks, blockSize>>>(
        csr_row_ptr, csr_col_ind, labels, changed, num_nodes
    );
    
    // Check for errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    return 0;
}

} // extern "C"