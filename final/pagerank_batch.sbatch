#!/bin/bash
#SBATCH --job-name=pagerank_benchmark
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint=gpu
#SBATCH --account=m4341_g
#SBATCH --qos=regular
#SBATCH --gpus-per-node=4

echo "Pagerank benchmark for only linalg"

VENV_PATH="/global/homes/a/asyam/myenv"
source $VENV_PATH/bin/activate

python scripts/run_page_alg.py \
    --gpu