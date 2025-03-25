#!/bin/bash
#SBATCH -C gpu                        
#SBATCH -N 1                          
#SBATCH --ntasks=1                    
#SBATCH --gpus-per-task=1            
#SBATCH -t 00:10:00                   
#SBATCH -J spmv-cuda-asyam            
#SBATCH --mail-user=as2839@cornell.edu # Email for notifications
#SBATCH --mail-type=ALL
#SBATCH -A m4341_g                     # Project account
#SBATCH -q debug                      # Use debug queue

# Set the matrix path.
# Modify the path below to point to your desired matrix.
export MATRIX_PATH="/global/homes/a/asyam/CS6230-group5/hw2-spmv/matrices/delaunay_n22/delaunay_n22.mtx"

# Load required modules
module load PrgEnv-gnu
module load cudatoolkit/12.4 

# (Optional) Print the matrix path for verification
echo "Running SpMV benchmark on matrix: $MATRIX_PATH"

# Run the test_spmv executable with the matrix path as an argument.
# This executable will first check correctness and then run benchmarks.
srun -n 1 -c 1 --cpu_bind=cores ./build/test_spmv $MATRIX_PATH