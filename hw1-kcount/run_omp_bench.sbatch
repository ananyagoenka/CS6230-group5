#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J kmer-omp-asyam
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00


DATASET=${KCOUNT_DATASET_PATH:-/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa}
echo "=== Parameter Test ==="
echo "Date: $(date)"
echo "Dataset: $DATASET"

# Test 1: Thread count impact
echo "--- Testing Thread Count Impact ---"
for threads in 1 2 4 8 16 32 64 128; do
    export KMER_IMPL=omp

    export OMP_NUM_THREADS=$threads
    export OMP_SCHEDULE="dynamic,64"
    export OMP_PROC_BIND=close
    echo "Running with $threads threads"
    srun -n 1 -c $threads --cpu_bind=cores ./kcount $DATASET
    echo ""
done

# Test 2: Chunk size impact (with fixed 64 threads)
echo "--- Testing Chunk Size Impact ---"
export OMP_NUM_THREADS=64
for chunk in 16 32 64 128 256; do
    export KMER_IMPL=omp
    export OMP_SCHEDULE="dynamic,$chunk"
    echo "Running with chunk size $chunk"
    srun -n 1 -c 64 --cpu_bind=cores ./kcount $DATASET
    echo ""
done

# Test 3: CPU binding impact (with fixed 64 threads)
echo "--- Testing CPU Binding Impact ---"
export KMER_IMPL=omp
export OMP_NUM_THREADS=64
export OMP_SCHEDULE="dynamic,64"

echo "Running with core binding"
export KMER_IMPL=omp
export OMP_PROC_BIND=close
export OMP_PLACES=cores
srun -n 1 -c 64 --cpu_bind=cores ./kcount $DATASET
echo ""

echo "Running with thread binding"
export KMER_IMPL=omp
export OMP_PROC_BIND=close
export OMP_PLACES=threads
srun -n 1 -c 64 --cpu_bind=threads ./kcount $DATASET
echo ""

echo "Running with socket binding"
export KMER_IMPL=omp
export OMP_PROC_BIND=spread
export OMP_PLACES=sockets
srun -n 1 -c 64 --cpu_bind=sockets ./kcount $DATASET
echo ""

echo "Running without explicit binding"
export KMER_IMPL=omp
unset OMP_PROC_BIND
unset OMP_PLACES
srun -n 1 -c 64 ./kcount $DATASET
echo ""
