#!/bin/bash
#SBATCH --job-name=kmer_omp
#SBATCH --output=kmer_omp.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --account=m4341  

# Load any necessary modules (if required on Perlmutter, e.g., for compilers)
# module load <module_name>

# Set the number of OpenMP threads to match the allocated CPUs
export OMP_NUM_THREADS=128

# Run the executable using srun with CPU binding.
# The environment variable KCOUNT_DATASET_PATH should be set to your dataset's path.
srun -C cpu -N 1 -n 1 -c 128 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH omp