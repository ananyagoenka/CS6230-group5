#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J kmer-hybrid-asyam
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00

module load gcc-native/13.2

# Set the implementation to OpenMP
export KMER_IMPL=hybrid

export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

echo "=== 8 MPI processes, 16 OpenMP threads each ==="
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun -n 8 -c 16 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
echo ""

# Test 2: Balanced MPI/OpenMP ratio
echo "=== 16 MPI processes, 8 OpenMP threads each ==="
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun -n 16 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
echo ""

# Test 3: More MPI processes, fewer OpenMP threads
echo "=== 32 MPI processes, 4 OpenMP threads each ==="
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun -n 32 -c 4 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
echo ""

# Test 4: Many MPI processes, few OpenMP threads
echo "=== 64 MPI processes, 2 OpenMP threads each ==="
export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun -n 64 -c 2 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH
echo ""