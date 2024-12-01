import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ScoreMatchingDenoisingModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        """
        A denoising score-matching network to predict the gradient of log-probability density.

        :param input_dim: Dimension of input features (e.g., x, y, degree, clustering)
        :param hidden_dim: Dimension of hidden layers
        :param output_dim: Dimension of the output (e.g., x, y)
        """
        super(ScoreMatchingDenoisingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Outputs gradients for x, y
        )

    def forward(self, noisy_features):
        """
        Forward pass: Predict the score (gradient of log-probability) for x, y.

        :param noisy_features: Noisy input features (n x input_dim tensor)
        :return: Predicted gradients for x, y (n x output_dim tensor)
        """
        return self.network(noisy_features)
    

class GCNScoreMatchingDenoisingModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        """
        A GCN-based denoising score matching model.

        :param input_dim: Dimension of input features (e.g., x, y, degree, clustering)
        :param hidden_dim: Dimension of hidden layers
        """
        super(GCNScoreMatchingDenoisingModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # Output is x, y coordinates

    def forward(self, x, edge_index):
        """
        Forward pass for the GCN model.

        :param x: Node features (n x input_dim tensor)
        :param edge_index: Edge index tensor (2 x num_edges)
        :return: Denoised x, y coordinates (n x 2 tensor)
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)  # Predict denoised x, y
        return x
