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

# Test 1: Varying number of tasks on 4 nodes
echo "=== Test 1: Varying number of tasks on 4 nodes ==="

echo "--- 4 nodes, 32 tasks (8 per node) ---"
srun -N 4 -n 32 -c 32 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

echo "--- 4 nodes, 64 tasks (16 per node) ---"
srun -N 4 -n 64 -c 16 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

echo "--- 4 nodes, 128 tasks (32 per node) ---"
srun -N 4 -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

echo "--- 4 nodes, 256 tasks (64 per node) ---"
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

# Test 2: Fixed number of tasks (128) spread across different node counts
echo "=== Test 2: Fixed tasks (128) with different node distributions ==="

echo "--- 1 node, 128 tasks ---"
srun -N 1 -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

echo "--- 2 nodes, 128 tasks (64 per node) ---"
srun -N 2 -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH

echo "--- 4 nodes, 128 tasks (32 per node) ---"
srun -N 4 -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH