import torch
import numpy as np
import scipy.sparse as sp
import cupy as cp
from torch_sparse import SparseTensor

def csr_to_cuda(csr_matrix):
    """
    Convert a scipy CSR matrix to cupy CSR matrix for GPU processing.
    
    Args:
        csr_matrix: Scipy CSR matrix
        
    Returns:
        cupy CSR matrix
    """
    return cp.sparse.csr_matrix(
        (cp.array(csr_matrix.data), 
         cp.array(csr_matrix.indices), 
         cp.array(csr_matrix.indptr)),
        shape=csr_matrix.shape
    )

def to_sparse_tensor(csr_matrix):
    """
    Convert a scipy CSR matrix to PyTorch sparse tensor.
    
    Args:
        csr_matrix: Scipy CSR matrix
        
    Returns:
        torch.sparse.FloatTensor
    """
    coo = csr_matrix.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = coo.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

def sparse_tensor_to_torch_geometric(sparse_tensor):
    """
    Convert PyTorch sparse tensor to torch_geometric SparseTensor.
    
    Args:
        sparse_tensor: PyTorch sparse tensor
        
    Returns:
        torch_geometric SparseTensor
    """
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    size = sparse_tensor.size()
    
    return SparseTensor(row=indices[0], col=indices[1], value=values, sparse_sizes=size)

def generate_sparse_random_matrix(n, density=0.01, device='cuda'):
    """
    Generate a random sparse matrix with given density.
    
    Args:
        n (int): Size of the square matrix
        density (float): Density of non-zero elements
        device (str): Device to place the tensor
        
    Returns:
        torch.sparse.FloatTensor
    """
    nnz = int(n * n * density)
    indices = torch.randint(0, n, (2, nnz), device=device)
    values = torch.randn(nnz, device=device)
    return torch.sparse.FloatTensor(indices, values, (n, n))

def is_symmetric(mat):
    """
    Check if a sparse matrix is symmetric.
    
    Args:
        mat: Sparse matrix (scipy, cupy, or PyTorch)
        
    Returns:
        bool: True if symmetric
    """
    if isinstance(mat, sp.spmatrix):
        # scipy sparse matrix
        return (mat != mat.T).nnz == 0
    elif isinstance(mat, cp.sparse.spmatrix):
        # cupy sparse matrix
        return (mat != mat.T).nnz == 0
    elif torch.is_tensor(mat) and mat.is_sparse:
        # PyTorch sparse tensor
        return torch.all(mat.to_dense() == mat.to_dense().T)
    else:
        raise ValueError("Unsupported matrix type")

def sparsity(mat):
    """
    Calculate the sparsity of a matrix (percentage of zeros).
    
    Args:
        mat: Matrix (scipy, cupy, PyTorch)
        
    Returns:
        float: Sparsity as a value between 0 and 1
    """
    if isinstance(mat, sp.spmatrix):
        # scipy sparse matrix
        return 1.0 - (mat.nnz / (mat.shape[0] * mat.shape[1]))
    elif isinstance(mat, cp.sparse.spmatrix):
        # cupy sparse matrix
        return 1.0 - (mat.nnz / (mat.shape[0] * mat.shape[1]))
    elif torch.is_tensor(mat):
        if mat.is_sparse:
            # PyTorch sparse tensor
            return 1.0 - (mat._nnz() / (mat.size(0) * mat.size(1)))
        else:
            # PyTorch dense tensor
            return (mat == 0).float().mean().item()
    else:
        raise ValueError("Unsupported matrix type")