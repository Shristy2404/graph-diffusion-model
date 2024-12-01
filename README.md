#### **Note** - This is a simplified, general version of the code I wrote for the actual project - which is proporietary and on company servers. I have shared as much as I can without violating CCI constraints. 

# Graph Diffusion Model
This repository demonstrates a diffusion-based approach to denoise noisy graph data using synthetic examples. The project includes graph generation, a denoising model, and an evaluation pipeline.

--- 
## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)

---

## Features

1. **Graph Generation**:
   - Generates random geometric graphs with meaningful node features:
     - \(x, y\) spatial coordinates.
     - Node degree.
     - Clustering coefficient.

2. **Noise Addition**:
   - Adds Gaussian noise to \(x, y\) coordinates of the graph nodes while preserving other features.

3. **Denoising Models**:
   - Implements two denoising models:
     - Fully connected network for score matching.
     - Graph Convolutional Network (GCN) for graph-aware denoising.

4. **Score Matching Loss**:
   - Uses the denoising score matching objective to predict gradients of log-probability density.

5. **Evaluation and Visualization**:
   - Computes Mean Squared Error (MSE) between clean and denoised coordinates.
   - Visualizes graphs (clean, noisy, and denoised) with side-by-side comparison.

---

## Repository Structure
- `data/`: Code for generating synthetic graphs and adding noise to node features
- `src/`: Core model, training, and evaluation scripts
- `utils/`: Helper functions for graph visualization and processing
- environment.yml
- requirements.txt
- LICENSE

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shristy2404/graph-diffusion-model.git
   cd graph-diffusion-model 
2. Create a virtual environment and install dependencies:

    ``` bash 
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
3. Install PyTorch and PyTorch Geometric:

    - Visit PyTorch and install the appropriate version.
    - Install PyTorch Geometric:

        ``` bash
        pip install torch-geometric

--- 

## Usage
1. Generate a random geometric graph with meaningful features:

    ```bash
    python data/generate_graph.py
2. Add Gaussian noise to the graph's x,y coordinates:

    ```bash
    python data/add_noise.py
3. Train the fully connected or GCN-based denoising model 
    ```bash
    python src/train.py
4. Evaluate the trained model using MSE as a metric:

    ```bash
    python evaluation/evaluate.py
5. Compare clean, noisy, and denoised graphs:

    ```bash
    python utils/graph_utils.py

---

## Dataset

The dataset is dynamically generated and stored in the data sample_graphs/ directory. It consists of:

- Clean Graph: Original graph with x,y coordinates and other features.
- Noisy Graph: Same graph with Gaussian noise added to x,y.

---

## Model
Fully Connected Denoising Model

- A simple feedforward network predicts gradients for x, y coordinates.
- Suitable for tasks with minimal reliance on graph structure.

GCN-Based Denoising Model

- Incorporates graph structure using two Graph Convolutional Layers (GCNs).
- Aggregates information from neighboring nodes for better denoising.