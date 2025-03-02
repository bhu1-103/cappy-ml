# Deep learning libraries.
import torch
import torch.nn.functional as F

import torch_geometric.data
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv, GATv2Conv, ClusterGCNConv, HypergraphConv, SuperGATConv, SAGEConv, TAGConv, \
    ARMAConv, HANConv, SplineConv, GraphConv, RGATConv, HEATConv


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, hiddens, n_targets):
        super().__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * n_node_features + n_edge_features, hiddens),
            torch.nn.BatchNorm1d(hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_features, hiddens, n_targets):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(n_node_features + hiddens, hiddens),
            torch.nn.BatchNorm1d(hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, hiddens),
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(n_node_features + hiddens, hiddens),
            torch.nn.BatchNorm1d(hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return out


# Completed
class MetaNet(torch.nn.Module):
    def __init__(self, n_node_features, n_edge_features, num_hidden):
        super(MetaNet, self).__init__()

        # Input Layer
        self.input = MetaLayer(
            edge_model=EdgeModel(
                n_node_features=n_node_features, n_edge_features=n_edge_features,
                hiddens=num_hidden, n_targets=num_hidden),
            node_model=NodeModel(n_node_features=n_node_features, hiddens=num_hidden, n_targets=num_hidden)

        )

        # Output Layer
        self.output = MetaLayer(
            edge_model=EdgeModel(
                n_node_features=num_hidden, n_edge_features=num_hidden,
                hiddens=num_hidden, n_targets=num_hidden),
            node_model=NodeModel(n_node_features=num_hidden, hiddens=num_hidden, n_targets=1)

        )

    def forward(self, data):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y

        x, edge_attr, _ = self.input(x, edge_index, edge_attr)
        x = F.relu(x)
        x, edge_attr, _ = self.output(x, edge_index, edge_attr)

        return x


class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, input_node_features, input_edge_features, hidden_dim, output_dim):
        super(GraphConvolutionalNetwork, self).__init__()

        # Graph convolution layers for nodes
        self.node_conv1 = GCNConv(input_node_features, hidden_dim)
        self.node_conv2 = GCNConv(hidden_dim, output_dim)

        # Linear transformation for edge features
        self.edge_linear = torch.nn.Linear(input_edge_features, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        # Apply the first graph convolution layer to node features
        x_node = self.node_conv1(x, edge_index)
        x_node = F.relu(x_node)
        # Apply the second graph convolution layer to node features
        x_node = self.node_conv2(x_node, edge_index)

        # Apply a linear transformation to edge features
        x_edge = self.edge_linear(edge_attr)

        # You can optionally add more layers or modify the architecture as needed

        return x_node


# Completed
class GAT(torch.nn.Module):
    # Graph Attention Network
    def __init__(self, dim_in, dim_out, dim_h, heads=8):
        super().__init__()
        # Input
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        # Output
        self.gat3 = GATv2Conv(dim_h * heads, 1, heads=1)
        self.optimizer = torch.optim.Adagrad(self.parameters(),
                                           lr=0.01,
                                           weight_decay=5e-4)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat3(h, edge_index)
        return h  # , F.log_softmax(h, dim=1)


# Completed
class TGAN(torch.nn.Module):
    # Self-Supervised Graph Attention Network
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = TAGConv(dim_in, dim_h)
        self.gcn2 = TAGConv(dim_h, 1)
        self.optimizer = torch.optim.AdamW(self.parameters(),
                                           lr=0.05,
                                           weight_decay=5e-4)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)

        h = F.dropout(h, p=0.5, training=self.training)

        h = self.gcn2(h, edge_index)
        return h

class GYAT(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_h, heads=8, dropout=0.6, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

    def compute_loss(self, predictions, labels):
        return F.cross_entropy(predictions, labels)

class ImprovedGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_h, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        
        self.gat2 = GATv2Conv(dim_h * heads, dim_h, heads=heads)
        
        self.gat3 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y

        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.relu(h)

        h = F.dropout(h, p=0.6, training=self.training)
        h_residual = h  # Residual connection
        h = self.gat2(h, edge_index)
        h = h + h_residual  # Add residual
        
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat3(h, edge_index)
        return h
