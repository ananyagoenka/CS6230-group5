#!/bin/bash
#SBATCH -N 4                  
#SBATCH -C cpu
#SBATCH -q debug              
#SBATCH -J kmer-mpi-asyam 
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00           
#SBATCH -A m4341           

export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

module load gcc-native/13.2

export KMER_IMPL=mpi
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Test with 1 node (32 tasks)
echo "=== Running with 1 node (32 tasks) ==="
srun -N 1 -n 32 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

# Test with 2 nodes (64 tasks)
echo "=== Running with 2 nodes (64 tasks) ==="
srun -N 2 -n 64 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

# Test with 4 nodes (128 tasks)
echo "=== Running with 4 nodes (128 tasks) ==="
srun -N 4 -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH