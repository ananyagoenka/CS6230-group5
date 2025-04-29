#!/bin/bash
#SBATCH --job-name=bfs_benchmark
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --account=m3958
#SBATCH --qos=regular

# Load modules
module load python
module load cudatoolkit
module load numpy
module load scipy

# Set up environment
export PYTHONPATH=$PWD:$PYTHONPATH

# Create output directory
mkdir -p results/bfs

# Generate synthetic data if it doesn't exist
if [ ! -d "data/synthetic/generated" ]; then
    echo "Generating synthetic data..."
    python data/synthetic/generate_graphs.py
fi

# Run BFS benchmark
echo "Running BFS benchmarks..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.algorithms.bfs import MatrixBFS
from src.utils.graph_io import load_sparse_matrix
import torch
import time

# Load graph
adj_matrix = load_sparse_matrix('data/synthetic/generated/rmat_large.npz')
num_nodes = adj_matrix.shape[0]
source_node = 0

# Initialize BFS
bfs = MatrixBFS(device='cuda')
bfs.preprocess(adj_matrix)

# Warmup
for _ in range(3):
    distances, predecessors = bfs.run(source_node, num_nodes)

# Benchmark
start = time.time()
distances, predecessors = bfs.run(source_node, num_nodes)
end = time.time()

print(f'BFS completed in {end - start:.6f} seconds')
print(f'Nodes reached: {torch.isfinite(distances).sum().item()}')
"

echo "BFS benchmark completed."