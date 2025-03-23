#!/bin/bash
#SBATCH -C gpu                        # Request GPU nodes
#SBATCH -N 1                          # Use 1 node
#SBATCH --ntasks=1                    # 1 MPI task
#SBATCH --cpus-per-task=1             # 1 CPU per task (since the work is on GPU)
#SBATCH -t 00:10:00                   # Wall clock time limit
#SBATCH -J spmv-cuda-asyam            # Job name
#SBATCH --mail-user=aag237@cornell.edu # Email for notifications
#SBATCH --mail-type=ALL
#SBATCH -A m4341                     # Project account
#SBATCH -q debug                      # Use debug queue

# Set the matrix path.
# Modify the path below to point to your desired matrix.
export MATRIX_PATH="/global/homes/a/ananyag/CS6230-group5/hw2-spmv/matrices/stomach/stomach.mtx"

# Load required modules
module load PrgEnv-gnu
module load cuda/11.7

# (Optional) Print the matrix path for verification
echo "Running SpMV benchmark on matrix: $MATRIX_PATH"

# Run the test_spmv executable with the matrix path as an argument.
# This executable will first check correctness and then run benchmarks.
srun -n 1 -c 1 --cpu_bind=cores ./test_spmv $MATRIX_PATH