#!/bin/bash

# This script sets up an environment for running benchmarks in an interactive session
# Usage: ./scripts/interactive.sh

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory
mkdir -p results

echo "Interactive environment ready"
echo "======================================================="
echo "To run BFS benchmarks on a small graph with GPU:"
echo "python scripts/run_bfs.py --sizes 100 500 --graph-type scale-free --gpu --plot"
echo ""
echo "To run PageRank benchmarks on a small graph with GPU:"
echo "python scripts/run_pagerank.py --sizes 100 500 --graph-type scale-free --gpu --plot"
echo ""
echo "To run all benchmarks with multiple graph types:"
echo "python scripts/run_benchmarks.py --sizes 100 500 1000 --graph-types scale-free random --gpu --plot"
echo "======================================================="

# Guide for requesting an interactive session
echo ""
echo "INTERACTIVE SESSION GUIDE:"
echo "To request an interactive session with a GPU, run:"
echo "salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --constraint=gpu_type:a100 --account=m3958 --qos=regular --time=02:00:00"
echo ""
echo "After getting the allocation, run this script to set up the environment."