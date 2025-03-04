#!/bin/bash
#SBATCH -N 4                  
#SBATCH -C cpu
#SBATCH -q debug              
#SBATCH -J kmer-mpi-asyam 
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00           
#SBATCH -A m4341           

# Set dataset path
export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Run the MPI K-mer counter
srun -n 128 -c 8 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH mpi
