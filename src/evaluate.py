import torch
from torch.utils.data import DataLoader
from data.generate_graph import generate_graph
from data.add_noise import add_noise_to_graph
from train import GraphDataset
from models import ScoreMatchingDenoisingModel

def mse(predictions, targets):
    """
    Compute Mean Squared Error (MSE) between predictions and targets.

    :param predictions: Tensor of predicted x, y coordinates (n x 2)
    :param targets: Tensor of ground truth x, y coordinates (n x 2)
    :return: Scalar MSE value
    """
    return torch.mean((predictions - targets) ** 2).item()

def evaluate_model(model, data_loader):
    """
    Evaluate the model on a dataset using MSE as the metric.

    :param model: The trained score matching model
    :param data_loader: DataLoader containing noisy and clean data
    :return: Average MSE over the dataset
    """
    model.eval()
    total_mse = 0
    with torch.no_grad():
        for noisy, clean in data_loader:
            predictions = model(noisy)  # Predict denoised x, y
            total_mse += mse(predictions, clean)
    avg_mse = total_mse / len(data_loader)
    return avg_mse

if __name__ == "__main__":

    # Generate a clean graph and noisy version
    clean_graph = generate_graph(n_nodes=10)
    noisy_graph = add_noise_to_graph(clean_graph, noise_std=0.1)

    # Prepare dataset and data loader
    dataset = GraphDataset(clean_graph, noisy_graph)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load or initialize the model
    model = ScoreMatchingDenoisingModel(input_dim=2)
    # Assume the model is already trained

    # Evaluate the model
    avg_mse = evaluate_model(model, loader)
    print(f"Average MSE on test set: {avg_mse:.4f}")
