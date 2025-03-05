#!/bin/bash
#SBATCH -J kmer-omp-asyam
#SBATCH --mail-user=as2839@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200GB
#SBATCH -o weak_scaling_%j.out

# Single dataset path
DATASET="${/global/homes/a/asyam/CS6230-group5/hw1-kcount/sample_input/reads.fa}"
if [ -z "$DATASET" ]; then
    echo "Error: KCOUNT_DATASET_PATH environment variable not set"
    exit 1
fi

echo "=== Weak Scaling Test with Single Dataset ==="
echo "Date: $(date)"
echo "Dataset: $DATASET"

# Create results directory
RESULTS_DIR="weak_scaling_results"
mkdir -p $RESULTS_DIR

# Function to run with specific thread count
run_test() {
    local threads=$1
    
    echo "---------------------------------------------"
    echo "Running with $threads threads"
    
    # Set environment variables
    export OMP_NUM_THREADS=$threads
    export OMP_SCHEDULE="dynamic,64"
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    
    # Run with appropriate thread count
    srun -n 1 -c $threads --cpu_bind=cores ./kcount $DATASET > "$RESULTS_DIR/run_${threads}.log" 2>&1
    
    # Extract timing information
    elapsed=$(grep "Elapsed time" "$RESULTS_DIR/run_${threads}.log" | awk '{print $3}')
    
    echo "Threads: $threads, Time: $elapsed seconds"
    echo "$threads,$elapsed" >> "$RESULTS_DIR/weak_scaling.csv"
    echo "---------------------------------------------"
    echo ""
}

# Create CSV header
echo "threads,time" > "$RESULTS_DIR/weak_scaling.csv"

# Run with different thread counts
run_test 1
run_test 2
run_test 4
run_test 8
run_test 16
run_test 32
run_test 64
run_test 128

# Generate summary with speedup calculations
echo "=== Weak Scaling Summary ==="
echo "Thread Count,Time (s),Speedup,Efficiency (%)" > "$RESULTS_DIR/summary.txt"

base_time=$(grep "^1," "$RESULTS_DIR/weak_scaling.csv" | cut -d, -f2)
for t in 1 2 4 8 16 32 64 128; do
    time=$(grep "^$t," "$RESULTS_DIR/weak_scaling.csv" | cut -d, -f2)
    
    # Calculate speedup and efficiency
    speedup=$(echo "scale=2; $base_time/$time" | bc)
    efficiency=$(echo "scale=2; 100*$speedup/$t" | bc)
    
    echo "$t,$time,$speedup,$efficiency" >> "$RESULTS_DIR/summary.txt"
    echo "Threads: $t, Time: $time s, Speedup: ${speedup}x, Efficiency: $efficiency%"
done

echo "=== Weak Scaling Test Complete ==="
echo "Results saved in $RESULTS_DIR/"