#!/bin/bash
#SBATCH -N 4                  
#SBATCH -C gpu
#SBATCH -q debug              
#SBATCH -J kmer_mpi_aag237   
#SBATCH --mail-user=aag237@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00           
#SBATCH -A m4341_g            

# Set dataset path
export KCOUNT_DATASET_PATH="/global/homes/a/ananyag/CS6230-group5/hw1-kcount/sample_input/reads.fa"

# Ensure OpenMP uses the right number of threads
export OMP_NUM_THREADS=4

module load PrgEnv-gnu
module load cray-mpich

# Run the MPI K-mer counter
srun --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH