import torch
from torch.utils.data import DataLoader, Dataset
from denoising_model import ScoreMatchingDenoisingModel
from data.generate_graph import generate_graph
from data.add_noise import add_noise_to_graph

class GraphDataset(Dataset):
    def __init__(self, clean_graph, noisy_graph):
        """
        Dataset that pairs noisy and clean node x, y coordinates.

        :param clean_graph: Clean NetworkX graph
        :param noisy_graph: Noisy NetworkX graph
        """
        self.clean_data = [
            torch.tensor(node_data['features'][:2], dtype=torch.float32)  # Clean x, y
            for _, node_data in clean_graph.nodes(data=True)
        ]
        self.noisy_data = [
            torch.tensor(node_data['features'][:2], dtype=torch.float32)  # Noisy x, y
            for _, node_data in noisy_graph.nodes(data=True)
        ]

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        return self.noisy_data[idx], self.clean_data[idx]


def score_matching_loss(model, noisy_features, clean_features, noise_std=0.1):
    """
    Compute the score matching loss for the denoising model.

    :param model: Score matching denoising model
    :param noisy_features: Noisy features (batch_size x 2)
    :param clean_features: Clean features (batch_size x 2)
    :param noise_std: Standard deviation of noise
    :return: Scalar loss value
    """
    predicted_score = model(noisy_features)  # Predict gradients
    target_score = -(noisy_features - clean_features) / (noise_std**2)
    loss = torch.mean(torch.sum((predicted_score - target_score)**2, dim=1))
    return loss

def train_model(model, train_loader, epochs=50, lr=1e-3, noise_std=0.1):
    """
    Train the denoising score matching model.

    :param model: The score matching denoising model
    :param train_loader: DataLoader for training data
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :param noise_std: Standard deviation of noise
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in train_loader:
            optimizer.zero_grad()
            loss = score_matching_loss(model, noisy, clean, noise_std)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":

    # Generate clean and noisy graphs
    clean_graph = generate_graph()
    noisy_graph = add_noise_to_graph(clean_graph)

    # Create dataset and dataloader
    dataset = GraphDataset(clean_graph, noisy_graph)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize and train the model
    model = ScoreMatchingDenoisingModel(input_dim=2)
    train_model(model, loader)
