import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import argparse
from Model import GAT  # Assuming GAT model is defined elsewhere
import sys
from torch_geometric.loader import DataLoader
import pickle
from torch_geometric.nn import GATConv

sys.path.append('C:/Users/galrt/PycharmProjects/SolarMLProject/DataProcessing')

# Function to load data
def load_data_list(filename, batch_size=32):
    with open(filename, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

# GAT model class definition
class GAT(nn.Module):
    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4, leaky_relu_slope=0.2):
        super(GAT, self).__init__()

        self.gat1 = GATConv(
            in_channels=in_features, out_channels=n_hidden, heads=n_heads,
            concat=concat, dropout=dropout, negative_slope=leaky_relu_slope
        )

        self.gat2 = GATConv(
            in_channels=n_hidden, out_channels=1, heads=1,
            concat=False, dropout=dropout, negative_slope=leaky_relu_slope
        )

    def forward(self, input_tensor: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        x = self.gat1(input_tensor, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x

# Training function
def train_iter(epoch, model, optimizer, criterion, data_loader, print_every=10):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_tensor = batch.x  # Node features
        edge_index = batch.edge_index  # Edge indices (graph structure)
        edge_attr = batch.edge_attr  # Edge attributes (optional)
        target = batch.y.squeeze()  # Flatten target tensor to match model output

        # Move tensors to the CPU (if not already on CPU)
        input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

        output = model(input_tensor, edge_index, edge_attr)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    if epoch % print_every == 0:
        print(f'Epoch: {epoch:04d} loss: {avg_loss:.4f}')

    return avg_loss

# Test function
def test(model, criterion, data_loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            input_tensor = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            target = batch.y.squeeze()  # Flatten target tensor to match model output

            input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

            output = model(input_tensor, edge_index, edge_attr)
            loss = criterion(output.squeeze(), target)

            total_loss += loss.item() * len(target)
            total_samples += len(target)

    avg_loss = total_loss / total_samples
    return avg_loss

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network for Regression')

    # Hyperparameters (you can modify these as needed)
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6, help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads (default: 8)')
    parser.add_argument('--concat-heads', action='store_true', default=False, help='whether to concatenate attention heads')
    parser.add_argument('--val-every', type=int, default=20, help='epochs to wait for print training and validation evaluation')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 16)')  # Set batch size to 16 to help with memory
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S', help='random seed (default: 13)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # This checks if the code should run on the GPU (CUDA-enabled device) or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using {device} device')

    # Load data
    start_time = time.time()
    load_data_list = load_data_list("C:/Users/hadar/Desktop/dataloader_pkl/loader_data_one_test_1.pkl", batch_size=16)  # Reduced batch size
    """"
    # Split data into train, validation, and test
    train_data = load_data_list[:int(0.7 * len(load_data_list))]
    val_data = load_data_list[int(0.7 * len(load_data_list)):int(0.85 * len(load_data_list))]
    test_data = load_data_list[int(0.85 * len(load_data_list)):]
    """""
    # Create data loaders
    data_loader_train = DataLoader(load_data_list, batch_size=args.batch_size, shuffle=True)
    data_loader_val = DataLoader(load_data_list, batch_size=args.batch_size, shuffle=False)
    data_loader_test = DataLoader(load_data_list, batch_size=args.batch_size, shuffle=False)

    # Model initialization
    Gat_net = GAT(
        in_features=3,  # Number of input features per node
        n_hidden=args.hidden_dim,  # Output size of the first Graph Attention Layer
        n_heads=args.num_heads,  # Number of attention heads in the first Graph Attention Layer
        num_classes=1,  # Only one output: GHI (regression task)
        concat=args.concat_heads,  # Whether to concatenate attention heads
        dropout=args.dropout_p,  # Dropout rate
        leaky_relu_slope=0.2  # Alpha (slope) of the leaky relu activation
    ).to(device)  # This moves the model to the CPU

    # Configure optimizer and loss function
    optimizer = Adam(Gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.MSELoss()

    print("Starting training and validation")

    # Train and evaluate the model
    for epoch in range(args.epochs):
        train_loss = train_iter(epoch + 1, Gat_net, optimizer, criterion, data_loader_train, args.val_every)

        if epoch % args.val_every == 0:
            val_loss = test(Gat_net, criterion, data_loader_val)
            print(f'Validation loss: {val_loss:.4f}')

        if args.dry_run:
            break

    end_time_3 = time.time()
    print(f"Program finished training and validation in {end_time_3 - start_time:.2f} seconds.")

    # Final test
    test_loss = test(Gat_net, criterion, data_loader_test)
    print(f'Test set results: loss {test_loss:.4f}')

    end_time_4 = time.time()
    print(f"Program finished testing in {end_time_4 - start_time:.2f} seconds.")
