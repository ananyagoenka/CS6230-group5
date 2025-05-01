#!/bin/bash
#SBATCH --job-name=graph_la_benchmarks
#SBATCH --output=graph_la_%j.out
#SBATCH --error=graph_la_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu_type:a100
#SBATCH --account=m3958
#SBATCH --qos=regular

# Load necessary modules (modify based on your system configuration)
module load python
module load cuda/11.7.0
module load cudnn/8.3.2

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory
mkdir -p results

# Run benchmarks
echo "Starting benchmarks at $(date)"

# You can adjust these parameters based on your needs
python scripts/run_benchmarks.py \
    --sizes 100 500 1000 5000 10000 \
    --graph-types scale-free random small-world \
    --runs 5 \
    --gpu \
    --plot \
    --save-dir results \
    --verbose

echo "Benchmarks completed at $(date)"