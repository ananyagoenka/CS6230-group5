import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, Reddit, Amazon
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import homophily
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sys
from collections import defaultdict, deque
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.bfs import BFS
from src.algorithms.pagerank import PageRank


# Check if running on Perlmutter
def is_perlmutter():
    """Check if code is running on NERSC Perlmutter"""
    return os.path.exists('/global/common/software/nersc')

# Configure for Perlmutter A100 GPUs if available
if is_perlmutter():
    print("Running on Perlmutter with A100 GPUs")
    # Set optimal PyTorch settings for A100 GPUs
    torch.backends.cudnn.benchmark = True
    # Perlmutter specific optimizations
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'  # Important for A100

# Helper functions for graph conversions
def pyg_to_adjacency_list(pyg_data):
    """Convert PyG graph to adjacency list"""
    edge_index = pyg_data.edge_index.cpu().numpy()
    adj_list = defaultdict(list)
    
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        adj_list[source].append(target)
    
    return adj_list

def pyg_to_adjacency_matrix(pyg_data, sparse=True, device=None):
    """Convert PyG graph to adjacency matrix with proper device handling
    
    Parameters:
    -----------
    pyg_data : torch_geometric.data.Data
        PyG data object
    sparse : bool
        Whether to return a sparse tensor
    device : torch.device or str
        Device to place the tensor on
        
    Returns:
    --------
    torch.Tensor or torch.sparse.Tensor
        Adjacency matrix
    """
    import torch
    
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes
    
    # Make sure edge_index is on the correct device
    if device is not None:
        edge_index = edge_index.to(device)
    
    if sparse:
        # Create sparse tensor with all components on the same device
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        adj_matrix = torch.sparse_coo_tensor(
            edge_index, 
            values,
            (num_nodes, num_nodes),
            device=edge_index.device
        )
        return adj_matrix
    else:
        # Create dense tensor
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix

def adjacency_list_to_edge_index(adj_list, num_nodes):
    """Convert adjacency list to edge_index format"""
    edges = []
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            edges.append((node, neighbor))
    
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

# Base GNN model
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_type='GCN'):
        super(GNN, self).__init__()
        
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# BFS Pipeline - Base class
class BFSPipeline:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def process_graph(self, data):
        """Process the graph with BFS algorithm - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train_epoch(self, data, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Process graph with BFS
        start_time = time.time()
        processed_data = self.process_graph(data)
        bfs_time = time.time() - start_time
        
        # Forward pass
        logits = self.model(processed_data.x, processed_data.edge_index)
        loss = criterion(logits[processed_data.train_mask], processed_data.y[processed_data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item(), bfs_time
    
    def evaluate(self, data):
        """Evaluate the model"""
        self.model.eval()
        
        # Process graph with BFS
        start_time = time.time()
        processed_data = self.process_graph(data)
        bfs_time = time.time() - start_time
        
        with torch.no_grad():
            logits = self.model(processed_data.x, processed_data.edge_index)
            pred = logits.argmax(dim=1)
            correct = (pred[processed_data.test_mask] == processed_data.y[processed_data.test_mask]).sum()
            acc = int(correct) / int(processed_data.test_mask.sum())
            
        return acc, bfs_time
    
    def run_pipeline(self, data, epochs=100, lr=0.01):
        """Run the complete pipeline"""
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        train_times, inference_times = [], []
        train_bfs_times, inference_bfs_times = [], []
        accuracies = []
        
        for epoch in range(epochs):
            start_time = time.time()
            loss, bfs_time = self.train_epoch(data, optimizer, criterion)
            train_time = time.time() - start_time
            
            start_time = time.time()
            acc, inference_bfs_time = self.evaluate(data)
            inference_time = time.time() - start_time
            
            train_times.append(train_time)
            inference_times.append(inference_time)
            train_bfs_times.append(bfs_time)
            inference_bfs_times.append(inference_bfs_time)
            accuracies.append(acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, '
                      f'Train Time: {train_time:.4f}s, Inference Time: {inference_time:.4f}s')
        
        return {
            'train_times': train_times,
            'inference_times': inference_times,
            'train_bfs_times': train_bfs_times,
            'inference_bfs_times': inference_bfs_times,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1]
        }

# BFS Pipeline - CPU Version
class BFSPipelineCPU(BFSPipeline):
    def process_graph(self, data):
        """Process graph with traditional CPU BFS"""
        # Convert to adjacency list
        adj_list = pyg_to_adjacency_list(data)
        
        # Apply BFS from a central node
        start_node = 0  # Can choose another node or sample multiple
        visited, distances = BFS.traditional_bfs_cpu(adj_list, start_node)
        
        # Create a new feature that encodes BFS distance as a feature channel
        max_distance = max([d for d in distances.values() if d != float('infinity')])
        distance_features = torch.zeros((data.num_nodes, 1), device=self.device)
        
        for node, dist in distances.items():
            if dist != float('infinity'):
                # Normalize distance
                distance_features[node, 0] = 1.0 - (dist / max_distance)
            # For unreachable nodes, leave as 0
        
        # Concatenate with original features
        enhanced_features = torch.cat([data.x, distance_features], dim=1)
        
        # Create a new Data object with enhanced features
        processed_data = Data(
            x=enhanced_features,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask
        )
        
        return processed_data

# BFS Pipeline - GPU Version
class BFSPipelineGPU(BFSPipeline):
    def process_graph(self, data):
        """Process graph with optimized GPU BFS"""
        # Convert to sparse adjacency matrix
        adj_matrix = pyg_to_adjacency_matrix(data, sparse=True, device=self.device)
        
        # Apply optimized sparse GPU BFS
        start_node = 0  # Can choose another node or sample multiple
        visited, distances = BFS.la_bfs_sparse_gpu_optimized_v2_turbo(adj_matrix, start_node)
        
        # Convert distances dict to tensor
        max_distance = max([d for d in distances.values() if d != float('infinity')])
        distance_features = torch.zeros((data.num_nodes, 1), device=self.device)
        
        for node, dist in distances.items():
            if dist != float('infinity'):
                # Normalize distance
                distance_features[node, 0] = 1.0 - (dist / max_distance)
        
        # Concatenate with original features
        enhanced_features = torch.cat([data.x, distance_features], dim=1)
        
        # Create a new Data object with enhanced features
        processed_data = Data(
            x=enhanced_features,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask
        )
        
        return processed_data

# PageRank Pipeline - Base class
class PageRankPipeline:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
    def process_graph(self, data):
        """Process the graph with PageRank algorithm - to be implemented by subclasses"""
        raise NotImplementedError
    
    def train_epoch(self, data, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        optimizer.zero_grad()
        
        # Process graph with PageRank
        start_time = time.time()
        processed_data = self.process_graph(data)
        pagerank_time = time.time() - start_time
        
        # Forward pass
        logits = self.model(processed_data.x, processed_data.edge_index)
        loss = criterion(logits[processed_data.train_mask], processed_data.y[processed_data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item(), pagerank_time
    
    def evaluate(self, data):
        """Evaluate the model"""
        self.model.eval()
        
        # Process graph with PageRank
        start_time = time.time()
        processed_data = self.process_graph(data)
        pagerank_time = time.time() - start_time
        
        with torch.no_grad():
            logits = self.model(processed_data.x, processed_data.edge_index)
            pred = logits.argmax(dim=1)
            correct = (pred[processed_data.test_mask] == processed_data.y[processed_data.test_mask]).sum()
            acc = int(correct) / int(processed_data.test_mask.sum())
            
        return acc, pagerank_time
    
    def run_pipeline(self, data, epochs=100, lr=0.01):
        """Run the complete pipeline"""
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        train_times, inference_times = [], []
        train_pr_times, inference_pr_times = [], []
        accuracies = []
        
        for epoch in range(epochs):
            start_time = time.time()
            loss, pr_time = self.train_epoch(data, optimizer, criterion)
            train_time = time.time() - start_time
            
            start_time = time.time()
            acc, inference_pr_time = self.evaluate(data)
            inference_time = time.time() - start_time
            
            train_times.append(train_time)
            inference_times.append(inference_time)
            train_pr_times.append(pr_time)
            inference_pr_times.append(inference_pr_time)
            accuracies.append(acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, '
                      f'Train Time: {train_time:.4f}s, Inference Time: {inference_time:.4f}s')
        
        return {
            'train_times': train_times,
            'inference_times': inference_times,
            'train_pr_times': train_pr_times,
            'inference_pr_times': inference_pr_times,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1]
        }

# PageRank Pipeline - CPU Version
class PageRankPipelineCPU(PageRankPipeline):
    def process_graph(self, data):
        """Process graph with traditional CPU PageRank"""
        # Convert to adjacency list
        adj_list = pyg_to_adjacency_list(data)
        
        # Apply PageRank
        pagerank_scores, _ = PageRank.traditional_pagerank_cpu(adj_list)
        
        # Create a new feature with PageRank scores
        pr_features = torch.zeros((data.num_nodes, 1), device=self.device)
        for node, score in pagerank_scores.items():
            pr_features[node, 0] = score
        
        # Normalize PageRank features
        pr_features = pr_features / pr_features.max()
        
        # Concatenate with original features
        enhanced_features = torch.cat([data.x, pr_features], dim=1)
        
        # Create a new Data object with enhanced features
        processed_data = Data(
            x=enhanced_features,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask
        )
        
        return processed_data

# PageRank Pipeline - GPU Version
class PageRankPipelineGPU(PageRankPipeline):
    def process_graph(self, data):
        """Process graph with optimized GPU PageRank"""
        # Convert to sparse adjacency matrix
        adj_matrix = pyg_to_adjacency_matrix(data, sparse=True, device=self.device)
        
        # Apply optimized sparse GPU PageRank
        pagerank_scores, _ = PageRank.la_pagerank_sparse_gpu_optimized_v2_turbo(adj_matrix)
        
        # Create a new feature with PageRank scores
        pr_features = pagerank_scores.unsqueeze(1)
        
        # Normalize PageRank features
        pr_features = pr_features / pr_features.max()
        
        # Concatenate with original features
        enhanced_features = torch.cat([data.x, pr_features], dim=1)
        
        # Create a new Data object with enhanced features
        processed_data = Data(
            x=enhanced_features,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask
        )
        
        return processed_data

# Main experiment function
def run_experiment(dataset_name='Cora', gnn_type='GCN', device='cuda', epochs=100):
    """Run the experiment with both CPU and GPU pipelines"""
    print(f"Running experiment with {dataset_name} dataset and {gnn_type} model")
    
    # Load the dataset
    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', transform=NormalizeFeatures())
    elif dataset_name == 'PubMed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=NormalizeFeatures())
    elif dataset_name == 'Reddit':
        dataset = Reddit(root='/tmp/Reddit')
    elif dataset_name == 'Amazon':
        dataset = Amazon(root='/tmp/Amazon', name='Computers')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data = dataset[0]
    
    # Create BFS models
    bfs_model_cpu = GNN(
        in_channels=dataset.num_features + 1,  # +1 for BFS distance feature
        hidden_channels=16,
        out_channels=dataset.num_classes,
        gnn_type=gnn_type
    )
    
    bfs_model_gpu = GNN(
        in_channels=dataset.num_features + 1,  # +1 for BFS distance feature
        hidden_channels=16,
        out_channels=dataset.num_classes,
        gnn_type=gnn_type
    )
    
    # Create PageRank models
    pr_model_cpu = GNN(
        in_channels=dataset.num_features + 1,  # +1 for PageRank score feature
        hidden_channels=16,
        out_channels=dataset.num_classes,
        gnn_type=gnn_type
    )
    
    pr_model_gpu = GNN(
        in_channels=dataset.num_features + 1,  # +1 for PageRank score feature
        hidden_channels=16,
        out_channels=dataset.num_classes,
        gnn_type=gnn_type
    )
    
    # Create pipelines
    bfs_pipeline_cpu = BFSPipelineCPU(bfs_model_cpu, device='cpu')
    bfs_pipeline_gpu = BFSPipelineGPU(bfs_model_gpu, device=device)
    
    pr_pipeline_cpu = PageRankPipelineCPU(pr_model_cpu, device='cpu')
    pr_pipeline_gpu = PageRankPipelineGPU(pr_model_gpu, device=device)
    
    # Run pipelines
    print("\nRunning BFS Pipeline (CPU version)...")
    bfs_cpu_results = bfs_pipeline_cpu.run_pipeline(data, epochs=epochs)
    
    print("\nRunning BFS Pipeline (GPU version)...")
    bfs_gpu_results = bfs_pipeline_gpu.run_pipeline(data, epochs=epochs)
    
    print("\nRunning PageRank Pipeline (CPU version)...")
    pr_cpu_results = pr_pipeline_cpu.run_pipeline(data, epochs=epochs)
    
    print("\nRunning PageRank Pipeline (GPU version)...")
    pr_gpu_results = pr_pipeline_gpu.run_pipeline(data, epochs=epochs)
    
    # Compute speedups
    bfs_train_speedup = np.mean(bfs_cpu_results['train_times']) / np.mean(bfs_gpu_results['train_times'])
    bfs_inference_speedup = np.mean(bfs_cpu_results['inference_times']) / np.mean(bfs_gpu_results['inference_times'])
    
    pr_train_speedup = np.mean(pr_cpu_results['train_times']) / np.mean(pr_gpu_results['train_times'])
    pr_inference_speedup = np.mean(pr_cpu_results['inference_times']) / np.mean(pr_gpu_results['inference_times'])
    
    # Print speedups
    print("\n" + "="*50)
    print(f"BFS Pipeline Speedups ({dataset_name}, {gnn_type}):")
    print(f"  Training Speedup: {bfs_train_speedup:.2f}x")
    print(f"  Inference Speedup: {bfs_inference_speedup:.2f}x")
    print(f"  BFS Computation Speedup: {np.mean(bfs_cpu_results['train_bfs_times']) / np.mean(bfs_gpu_results['train_bfs_times']):.2f}x")
    print(f"  Final Accuracy (CPU): {bfs_cpu_results['final_accuracy']:.4f}")
    print(f"  Final Accuracy (GPU): {bfs_gpu_results['final_accuracy']:.4f}")
    
    print("\n" + "="*50)
    print(f"PageRank Pipeline Speedups ({dataset_name}, {gnn_type}):")
    print(f"  Training Speedup: {pr_train_speedup:.2f}x")
    print(f"  Inference Speedup: {pr_inference_speedup:.2f}x")
    print(f"  PageRank Computation Speedup: {np.mean(pr_cpu_results['train_pr_times']) / np.mean(pr_gpu_results['train_pr_times']):.2f}x")
    print(f"  Final Accuracy (CPU): {pr_cpu_results['final_accuracy']:.4f}")
    print(f"  Final Accuracy (GPU): {pr_gpu_results['final_accuracy']:.4f}")
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # BFS training time
    plt.subplot(2, 2, 1)
    plt.plot(bfs_cpu_results['train_times'], label='CPU', marker='o', markersize=3)
    plt.plot(bfs_gpu_results['train_times'], label='GPU', marker='x', markersize=3)
    plt.title('BFS Pipeline: Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    # PageRank training time
    plt.subplot(2, 2, 2)
    plt.plot(pr_cpu_results['train_times'], label='CPU', marker='o', markersize=3)
    plt.plot(pr_gpu_results['train_times'], label='GPU', marker='x', markersize=3)
    plt.title('PageRank Pipeline: Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    # BFS accuracy
    plt.subplot(2, 2, 3)
    plt.plot(bfs_cpu_results['accuracies'], label='CPU', marker='o', markersize=3)
    plt.plot(bfs_gpu_results['accuracies'], label='GPU', marker='x', markersize=3)
    plt.title('BFS Pipeline: Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # PageRank accuracy
    plt.subplot(2, 2, 4)
    plt.plot(pr_cpu_results['accuracies'], label='CPU', marker='o', markersize=3)
    plt.plot(pr_gpu_results['accuracies'], label='GPU', marker='x', markersize=3)
    plt.title('PageRank Pipeline: Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_{gnn_type}_results.png')
    print(f"Results plot saved to {dataset_name}_{gnn_type}_results.png")
    
    return {
        'bfs_cpu': bfs_cpu_results,
        'bfs_gpu': bfs_gpu_results,
        'pr_cpu': pr_cpu_results,
        'pr_gpu': pr_gpu_results,
        'bfs_train_speedup': bfs_train_speedup,
        'bfs_inference_speedup': bfs_inference_speedup,
        'pr_train_speedup': pr_train_speedup,
        'pr_inference_speedup': pr_inference_speedup,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN Pipeline with BFS/PageRank')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'CiteSeer', 'PubMed', 'Reddit', 'Amazon'],
                        help='Dataset to use')
    parser.add_argument('--gnn', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'],
                        help='GNN model to use')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU only')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = run_experiment(
        dataset_name=args.dataset,
        gnn_type=args.gnn,
        device=device,
        epochs=args.epochs
    )