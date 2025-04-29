import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
from scipy import sparse

def sparse_matrix_to_edge_index(adj_matrix):
    """
    Convert a sparse adjacency matrix to edge index format for PyG.
    
    Args:
        adj_matrix: Sparse adjacency matrix (scipy or torch)
        
    Returns:
        torch.Tensor: Edge index tensor
    """
    if isinstance(adj_matrix, sparse.spmatrix):
        # Convert scipy sparse matrix to edge index
        coo = adj_matrix.tocoo()
        indices = np.vstack((coo.row, coo.col))
        edge_index = torch.tensor(indices, dtype=torch.long)
    elif torch.is_tensor(adj_matrix) and adj_matrix.is_sparse:
        # Convert torch sparse tensor to edge index
        indices = adj_matrix._indices()
        edge_index = indices
    else:
        raise ValueError("Input must be a scipy sparse matrix or torch sparse tensor")
        
    return edge_index

def create_pyg_graph(adj_matrix, features=None, labels=None):
    """
    Create a PyG graph from adjacency matrix and node features.
    
    Args:
        adj_matrix: Sparse adjacency matrix
        features: Node feature matrix (optional)
        labels: Node labels (optional)
        
    Returns:
        torch_geometric.data.Data: PyG graph
    """
    # Convert to edge index
    edge_index = sparse_matrix_to_edge_index(adj_matrix)
    
    # Create node features if not provided
    if features is None:
        num_nodes = adj_matrix.shape[0]
        features = torch.ones((num_nodes, 1), dtype=torch.float)
    
    # Create PyG graph
    data = Data(x=features, edge_index=edge_index)
    
    # Add labels if provided
    if labels is not None:
        data.y = labels
        
    return data

class OptimizedGCN(torch.nn.Module):
    """
    GCN with optimized graph operations using linear algebra.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialize GCN.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output features
        """
        super(OptimizedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge index
            
        Returns:
            torch.Tensor: Node embeddings
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings.
        
        Args:
            x: Node features
            edge_index: Edge index
            
        Returns:
            torch.Tensor: Node embeddings
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        return x

class OptimizedGraphSAGE(torch.nn.Module):
    """
    GraphSAGE with optimized graph operations using linear algebra.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialize GraphSAGE.
        
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output features
        """
        super(OptimizedGraphSAGE, self).__init__()
        self.conv1 = pyg.nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = pyg.nn.SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge index
            
        Returns:
            torch.Tensor: Node embeddings
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_gnn(model, data, optimizer, criterion, num_epochs=100):
    """
    Train GNN model.
    
    Args:
        model: GNN model
        data: PyG graph data
        optimizer: Optimizer
        criterion: Loss function
        num_epochs (int): Number of training epochs
        
    Returns:
        list: Training losses
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch + 1:03d}, Loss: {loss.item():.4f}')
    
    return losses

def test_gnn(model, data):
    """
    Test GNN model.
    
    Args:
        model: GNN model
        data: PyG graph data
        
    Returns:
        float: Test accuracy
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc