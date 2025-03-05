#!/bin/bash
#SBATCH -C cpu               
#SBATCH -N 1                 
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=1     
#SBATCH -t 00:10:00
#SBATCH -J kmer-serial-asyam
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -A m4341
#SBATCH -q debug

#MODIFY THE DATASET PATH TO RUN ON A DIFFERENT DATASET
export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

module load PrgEnv-gnu
module load gcc-native/13.2

export KMER_IMPL=serial
export MPICH_MPIIO_DVS_MAXNODES=1

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 1 -c 1 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH 
