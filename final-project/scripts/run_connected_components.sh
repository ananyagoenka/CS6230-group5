# Run Connected Components benchmark
echo "Running Connected Components benchmarks..."
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.algorithms.connected_components import MatrixConnectedComponents
from src.utils.graph_io import load_sparse_matrix
import torch
import time

# Load graph
adj_matrix = load_sparse_matrix('data/synthetic/generated/rmat_large.npz')
num_nodes = adj_matrix.shape[0]

# Initialize Connected Components
cc = MatrixConnectedComponents(device='cuda')
cc.preprocess(adj_matrix)

# Warmup
for _ in range(3):
    components = cc.run(num_nodes)

# Benchmark
start = time.time()
components = cc.run(num_nodes)
end = time.time()

print(f'Connected Components completed in {end - start:.6f} seconds')
print(f'Number of components: {len(torch.unique(components))}')
"

echo "Connected Components benchmark completed."