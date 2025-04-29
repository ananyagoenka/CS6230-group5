#!/bin/bash
#SBATCH --job-name=pagerank_benchmark
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
mkdir -p results/pagerank

# Generate synthetic data if it doesn't exist
if [ ! -d "data/synthetic/generated" ]; then
    echo "Generating synthetic data..."
    python data/synthetic/generate_graphs.py
fi

# Run PageRank benchmark
echo "Running PageRank benchmarks..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.algorithms.pagerank import MatrixPageRank
from src.utils.graph_io import load_sparse_matrix
import torch
import time

# Load graph
adj_matrix = load_sparse_matrix('data/synthetic/generated/rmat_large.npz')
num_nodes = adj_matrix.shape[0]

# Initialize PageRank
pr = MatrixPageRank(device='cuda')
pr.preprocess(adj_matrix)

# Warmup
for _ in range(3):
    scores = pr.run(num_nodes)

# Benchmark
start = time.time()
scores = pr.run(num_nodes)
end = time.time()

print(f'PageRank completed in {end - start:.6f} seconds')
print(f'Sum of scores: {scores.sum().item():.6f}')
"

echo "PageRank benchmark completed."