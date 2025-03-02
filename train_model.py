import argparse
import os

from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from dataset import CAPW19
from model import MetaNet
from evaluation import scores
from model import *



########################################################################################################################
# Weights-and-Biases logging.
########################################################################################################################

'''import wandb
wandb.init(project="GCN")'''

########################################################################################################################
# Command line arguments.
########################################################################################################################

parser = argparse.ArgumentParser(description='Pre-process and split the raw graphs dataset.')

parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs.')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size.')
parser.add_argument('--learning-rate', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay.')
parser.add_argument('--log-interval', default=1, type=int, help='Logging interval.')
parser.add_argument('--checkpoint-interval', default=1, type=int, help='Checkpoint interval.')
parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory.')

args = parser.parse_args()


########################################################################################################################
# Log parameters.
########################################################################################################################

'''wandb.config.epochs = args.epochs
wandb.config.batch_size = args.batch_size
wandb.config.learning_rate = args.learning_rate
wandb.config.weight_decay = args.weight_decay'''

########################################################################################################################
# Dataset.
########################################################################################################################

# Load training dataset.
dataset_train = CAPW19('./datasets/CAPW19/', split='train')
dataset_valid = CAPW19('./datasets/CAPW19/', split='valid')

# Dataset loaders.
train_loader = DataLoader(dataset_train, batch_size=args.batch_size)
valid_loader = DataLoader(dataset_valid, batch_size=1)

# Extract graph input sizes
#num_node_features = data.x.size(1)  # Number of node features
#num_edge_features = data.edge_attr.size(1) if data.edge_attr is not None else 0  # Edge features (if applicable)

# Extract graph input sizes from dataset
data = dataset_train[0]  # Get the first graph in the dataset to determine feature sizes
num_node_features = data.x.size(1)  # Number of node features
num_edge_features = data.edge_attr.size(1) if data.edge_attr is not None else 0  # Number of edge features, if available

# Define hyperparameters
num_hidden = 64  # Hidden layer size
dim_out = 1      # Output dimension (e.g., 1 for regression tasks)
heads = 8        # Number of GAT attention heads

########################################################################################################################
# Device setup.
########################################################################################################################

# Compute device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

########################################################################################################################
# Model.
########################################################################################################################

# Network configuration.
num_node_features = dataset_train[0].x.shape[1]
num_edge_features = dataset_train[0].edge_attr.shape[1]
num_hidden = 256


#wandb.config.num_hidden = num_hidden

# Create model.
#model = MetaNet(num_node_features, num_edge_features, num_hidden).to(device)
#model = TGAN(num_node_features, num_edge_features, num_hidden).to(device)
#model = GAT(num_node_features, num_edge_features, num_hidden).to(device)
model = ImprovedGAT(num_node_features, num_edge_features, num_hidden).to(device)
#model = GYAT(dim_in=num_node_features, dim_out=dim_out, dim_h=num_hidden, heads=heads).to(device)

# Monitor gradients and record the graph structure (+-).2
#wandb.watch(model)

########################################################################################################################
# Training utilities.
########################################################################################################################


def train(dataset):
    # Monitor training.
    losses = []

    # Put model in training mode!
    model.train()
    for batch in dataset:
        # Training step.
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        #loss = torch.sqrt(F.mse_loss(out.squeeze()[batch.y_mask], batch.y[batch.y_mask]))
        loss = torch.sqrt(F.mse_loss(out.squeeze(-1)[batch.y_mask], batch.y[batch.y_mask]))
        loss.backward()
        optimizer.step()
        # Monitoring
        losses.append(loss.item())

    # Return training metrics.
    return losses


def evaluate(dataset):
    # Monitor evaluation.
    losses = []
    rmse = []

    # Validation (1)
    model.eval()
    for batch in dataset:
        batch = batch.to(device)

        # Calculate validation losses.
        out = model(batch)
        loss = torch.sqrt(F.mse_loss(out.squeeze()[batch.y_mask], batch.y[batch.y_mask]))

        rmse_batch = scores(batch, out)

        # Metric logging.
        losses.append(loss.item())
        rmse.append(rmse_batch.item())

    return losses, rmse

########################################################################################################################
# Training loop.
########################################################################################################################

# Configuration
NUM_EPOCHS = args.epochs
LOG_INTERVAL = args.log_interval
CHECKPOINT_INTERVAL = args.checkpoint_interval
CHECKPOINT_DIR = args.checkpoint_dir

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# Configure optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# Metrics recorder per epoch.
train_losses = []

valid_losses = []
valid_losses_corrected = []

# Training loop.
model.train()
for epoch in range(NUM_EPOCHS):
    # Train.
    train_epoch_losses = train(train_loader)
    valid_epoch_losses, valid_epoch_losses_corrected = evaluate(valid_loader)

    # Log training metrics.
    train_avg_loss = np.mean(train_epoch_losses)
    train_losses.append(train_avg_loss)

    # Log validation metrics.
    valid_avg_loss = np.mean(valid_epoch_losses)
    valid_losses.append(valid_avg_loss)

    valid_avg_loss_corrected = np.mean(valid_epoch_losses_corrected)
    valid_losses_corrected.append(valid_avg_loss_corrected)

    #wandb.log({'epoch': epoch, 'train_loss': train_avg_loss, 'valid_loss': valid_avg_loss, 'score': valid_avg_loss_corrected})
    if epoch % LOG_INTERVAL == 0:
        print(f"epoch={epoch}, train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, valid_loss*={valid_avg_loss_corrected}")

    if epoch % CHECKPOINT_INTERVAL == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_avg_loss,
        }

        checkpoint_fn = os.path.join(CHECKPOINT_DIR, f'checkpoint-{epoch}.tar')
        torch.save(checkpoint, checkpoint_fn)
        #wandb.save(checkpoint_fn)

path = r"./pls-work/output"+ str(num_hidden) + str("_model.pt")
torch.save({
    'epoch':epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'loss':train_avg_loss},path)
