import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from Model import GAT  # Make sure you have the GAT model in the 'Model' directory.
import sys
from DataProcessing.Converting_Dataset_to_DataLoader import create_data_loader_from_directory

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.append('C:/Users/galrt/PycharmProjects/SolarMLProject/DataProcessing')


class GAT(nn.Module):
    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4, leaky_relu_slope=0.2):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, negative_slope=leaky_relu_slope
        )

        self.gat2 = GATConv(
            in_features=n_hidden, out_features=1, n_heads=1,
            concat=False, dropout=dropout, negative_slope=leaky_relu_slope
        )

    def forward(self, input_tensor: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        x = self.gat1(input_tensor, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x

def train_iter(epoch, model, optimizer, criterion, data_loader, print_every=10):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_tensor, edge_index, edge_attr, target = batch
        input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

        output = model(input_tensor, edge_index, edge_attr)
        loss = criterion(output.squeeze(), target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    if epoch % print_every == 0:
        print(f'Epoch: {epoch:04d}, Loss: {avg_loss:.4f}')

    return avg_loss


def test(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            input_tensor, edge_index, edge_attr, target = batch
            input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

            output = model(input_tensor, edge_index, edge_attr)
            loss = criterion(output.squeeze(), target)

            total_loss += loss.item() * len(target)
            total_samples += len(target)

    avg_loss = total_loss / total_samples
    return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network for Regression')

    # Command-line arguments
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6, help='Dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads (default: 8)')
    parser.add_argument('--concat-heads', action='store_true', default=False, help='Whether to concatenate attention heads')
    parser.add_argument('--val-every', type=int, default=20, help='Epochs to wait for print training and validation evaluation')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S', help='Random seed (default: 13)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Check if CUDA is available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using {device} device')

    start_time = time.time()

    # Load data
    load_data = create_data_loader_from_directory("C:/Users/galrt/Desktop/data")
    print(load_data)

    data_loader_train = DataLoader(load_data('train'), batch_size=args.batch_size, shuffle=True)
    data_loader_val = DataLoader(load_data('val'), batch_size=args.batch_size, shuffle=False)
    data_loader_test = DataLoader(load_data('test'), batch_size=args.batch_size, shuffle=False)

    end_time_1 = time.time()
    print(f"Data loading finished in {end_time_1 - start_time:.2f} seconds.")

    # Create the GAT model
    Gat_net = GAT(
        in_features=input_tensor.shape[1],  # Number of input features per node
        n_hidden=args.hidden_dim,  # Output size of the first Graph Attention Layer
        n_heads=args.num_heads,  # Number of attention heads in the first Graph Attention Layer
        num_classes=1,  # Only one output: GHI (regression task)
        concat=args.concat_heads,  # Whether to concatenate attention heads
        dropout=args.dropout_p,  # Dropout rate
        leaky_relu_slope=0.2  # Alpha (slope) of the leaky relu activation
    ).to(device)

    # Optimizer and loss function
    optimizer = Adam(Gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.MSELoss()

    print("Starting training and validation")

    for epoch in range(args.epochs):
        train_loss = train_iter(epoch + 1, Gat_net, optimizer, criterion, data_loader_train, args.val_every)

        if epoch % args.val_every == 0:
            val_loss = test(Gat_net, criterion, data_loader_val)
            print(f'Validation loss: {val_loss:.4f}')

        if args.dry_run:
            break

    end_time_3 = time.time()
    print(f"Training and validation finished in {end_time_3 - start_time:.2f} seconds.")

    # Final test
    test_loss = test(Gat_net, criterion, data_loader_test)
    print(f'Test set results: loss {test_loss:.4f}')

    end_time_4 = time.time()
    print(f"Testing finished in {end_time_4 - start_time:.2f} seconds.")
