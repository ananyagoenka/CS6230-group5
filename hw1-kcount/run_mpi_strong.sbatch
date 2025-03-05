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

# Test node count scaling with 64 processes per node
echo "=== Node Count Scaling (64 processes per node) ==="

for nodes in 1 2 3 4; do
    tasks=$((nodes * 64))
    cores=$((128 / 64))  # Assuming 128 cores per node on Perlmutter
    
    echo "--- $nodes node(s), $tasks total tasks (64 per node), $cores cores per task ---"
    srun -N $nodes -n $tasks -c $cores --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
    echo ""
done