#!/bin/bash
#SBATCH -C cpu               
#SBATCH -N 1                 
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=1     
#SBATCH --time=00:10:00       
#SBATCH -J kmer_serial_as2839
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4341

module load PrgEnv-gnu

export KCOUNT_DATASET_PATH="/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa"

# Run the serial K-mer counter (no MPI, no OpenMP)
srun ./kcount $KCOUNT_DATASET_PATH serial
