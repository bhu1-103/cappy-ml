import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import CAPW19
from model import *
from evaluation import access_point_throughputs
import wandb

# Initialize Weights-and-Biases logging
wandb.init(project="throughput_predfff")

# Command line arguments
parser = argparse.ArgumentParser(description='Pre-process and split the raw graphs dataset.')
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size.')
parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay.')
parser.add_argument('--log-interval', default=1, type=int, help='Logging interval.')
parser.add_argument('--checkpoint-interval', default=1, type=int, help='Checkpoint interval.')
parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory.')

args = parser.parse_args()

device = torch.device('cpu')
print(f'Using device: {device}')

dataset_test = CAPW19('./datasets/CAPW19/', split='test')

# Dataset loader
test_loader = DataLoader(dataset_test, batch_size=1)

num_node_features = dataset_test[0].x.shape[1]
num_edge_features = dataset_test[0].edge_attr.shape[1]
num_hidden = 256


def make_predictions(path2, predictions_loc, dataset):
    # Initialize model
    model = GAT(num_node_features, num_edge_features, num_hidden).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load model checkpoint
    #checkpoint = torch.load(path2)
    checkpoint = torch.load(path2, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Track metrics
    losses = []
    rmse_values = []

    # Validation loop
    model.eval()
    for batch in dataset:
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch)

        # Calculate predictions and losses for access points
        station_predictions = out[batch.y_mask]
        station_labels = batch.node_ap[batch.y_mask]
        ap_predictions = access_point_throughputs(station_predictions, station_labels).to(device)
        out[~batch.y_mask] = ap_predictions

        # Calculate loss (e.g., MSE) and RMSE for the batch
        batch_loss = F.mse_loss(out[batch.y_mask], station_labels)
        batch_rmse = torch.sqrt(batch_loss)
        losses.append(batch_loss.item())
        rmse_values.append(batch_rmse.item())

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'loss': batch_loss.item(),
            'rmse': batch_rmse.item()
        })

        # Save predictions to CSV
        preds_sta = out.detach().numpy()
        deployment = batch.deployment.detach().numpy()[0]
        scenario = batch.scenario[0]

        predict_loc = os.path.join(predictions_loc, scenario)
        os.makedirs(predict_loc, exist_ok=True)
        predict_fn = f'throughput_{deployment}.csv'
        predict_loc = os.path.join(predict_loc, predict_fn)
        df = pd.DataFrame(data=preds_sta, columns=['thr_pred'])
        print(predict_loc)
        df.to_csv(predict_loc, index=False)

    return


# Specify paths
loc = r"../kappi/resultsf/throupt_results/"
path2 = r"./pls-work/output256_model.pt"
make_predictions(path2, loc, test_loader)

# End the wandb run
wandb.finish()
