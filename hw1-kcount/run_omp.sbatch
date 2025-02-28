#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -J kmer_omp_as2839
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4341

#OpenMP settings:

# export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

# export OMP_NUM_THREADS=32
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

# #run the application:
# #applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
# srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  /global/homes/a/asyam/CS6230-group5/hw1-kcount/kcount

module load PrgEnv-gnu

export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

export OMP_NUM_THREADS=128   
export OMP_PLACES=threads    
export OMP_PROC_BIND=spread  

# Run the OpenMP K-mer counter with proper CPU binding
srun -n 1 -c 128 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH omp