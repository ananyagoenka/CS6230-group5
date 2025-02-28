#!/bin/bash
#SBATCH -J kmer_mpi            # Job name
#SBATCH -C cpu                 # Use CPU nodes
#SBATCH -N 4                   # Number of nodes (max 4 as per requirements)
#SBATCH -n 128                 # Number of MPI tasks (32 tasks per node)
#SBATCH -t 00:30:00            # Time limit (30 minutes)
#SBATCH --cpu_bind=cores       # Bind processes to cores

# Load required modules
module load PrgEnv-gnu

# Run the MPI application with optimal parameters
# The dataset path is read from the environment variable
srun -n 128 --cpu_bind=cores ./kcount $KCOUNT_DATASET_PATH