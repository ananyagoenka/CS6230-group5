#!/bin/bash
#SBATCH --job-name=graph_benchmarks
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
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

# Create output directories
mkdir -p results/strong_scaling
mkdir -p results/weak_scaling
mkdir -p results/sparsity_impact
mkdir -p results/ml_benchmarks

# Generate synthetic data
echo "Generating synthetic data..."
python data/synthetic/generate_graphs.py

# Optional: Download real-world datasets
# python data/real/download_datasets.py

# Run benchmarks
echo "Running strong scaling benchmarks..."
python benchmarks/strong_scaling.py

echo "Running weak scaling benchmarks..."
python benchmarks/weak_scaling.py

echo "Running sparsity impact benchmarks..."
python benchmarks/sparsity_impact.py

echo "Running ML integration benchmarks..."
python benchmarks/ml_benchmarks.py

echo "All benchmarks completed."