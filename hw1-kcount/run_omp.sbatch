#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -J kmer_omp_aag237
#SBATCH --mail-user=aag237@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4341_g

#OpenMP settings:

export KCOUNT_DATASET_PATH="/global/homes/a/ananyag/CS6230-group5/hw1-kcount/sample_input/reads.fa"
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1  /global/homes/a/ananyag/CS6230-group5/hw1-kcount/kcount
