import torch
import os
from contextlib import contextmanager

def get_available_gpus():
    """
    Get the number of available GPUs.
    
    Returns:
        int: Number of available GPUs
    """
    return torch.cuda.device_count()

def get_gpu_memory_usage():
    """
    Get the current GPU memory usage for all GPUs.
    
    Returns:
        list: List of (allocated, cached) memory in MB for each GPU
    """
    memory_usage = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
        cached = torch.cuda.memory_reserved(i) / (1024 * 1024)
        memory_usage.append((allocated, cached))
    return memory_usage

@contextmanager
def gpu_timer(device=None):
    """
    Context manager for timing GPU operations.
    
    Args:
        device: CUDA device to synchronize with
        
    Yields:
        None
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    yield
    end.record()
    
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / 1000  # Convert to seconds

def set_gpu_for_process(rank, world_size):
    """
    Set up the GPU for the current process in a multi-GPU setting.
    
    Args:
        rank (int): Rank of the current process
        world_size (int): Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set device
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    """
    Clean up distributed training resources.
    """
    torch.distributed.destroy_process_group()

def get_perlmutter_gpu_config():
    """
    Get the configuration for Perlmutter GPUs.
    
    Returns:
        dict: Configuration details for Perlmutter GPUs
    """
    return {
        'gpu_type': 'NVIDIA A100',
        'gpu_memory': 40, # GB per GPU
        'gpus_per_node': 4,
        'gpu_arch': 'Ampere',
        'cuda_capability': '8.0',
        'bandwidth': 1555, # GB/s (HBM2e)
        'tensor_cores': True,
        'nvlink_bandwidth': 600, # GB/s
        'pcie_bandwidth': 64, # GB/s (PCIe Gen4 x16)
    }