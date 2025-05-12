#!/bin/bash
#SBATCH --job-name=bfs_benchmark
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint=gpu
#SBATCH --account=m4341_g
#SBATCH --qos=regular
#SBATCH --gpus-per-node=4

VENV_PATH="/global/homes/a/asyam/myenv"
source $VENV_PATH/bin/activate

python scripts/run_bfs.py \
    --gpu