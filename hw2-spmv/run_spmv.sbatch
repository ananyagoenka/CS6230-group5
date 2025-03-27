#!/bin/bash
#SBATCH -C gpu                        
#SBATCH -N 1                          
#SBATCH --ntasks=1                    
#SBATCH --gpus-per-task=1            
#SBATCH -t 00:10:00                   
#SBATCH -J spmv-cuda-asyam            
#SBATCH --mail-user=as2839@cornell.edu 
#SBATCH --mail-type=ALL
#SBATCH -A m4341_g                     
#SBATCH -q debug                      

# Choose/modify the path below to point to desired matrix
#n16 matrix
export MATRIX_PATH="/global/homes/a/asyam/CS6230-group5/hw2-spmv/matrices/n16.mtx"

#stomach matrix
# export MATRIX_PATH="/global/homes/a/asyam/CS6230-group5/hw2-spmv/matrices/stomach/stomach.mtx"

#n22 delaunay matrix
# export MATRIX_PATH="/global/homes/a/asyam/CS6230-group5/hw2-spmv/matrices/delaunay_n22/delaunay_n22.mtx"

# Load required modules
module load PrgEnv-gnu
module load cudatoolkit/12.4 

echo "Running SpMV benchmark on matrix: $MATRIX_PATH"

srun -n 1 -c 1 --cpu_bind=cores ./test_spmv $MATRIX_PATH