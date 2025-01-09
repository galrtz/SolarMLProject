import os  # Importing the os library for handling file paths and system operations
import time  # Importing time for tracking execution time
import torch  # Importing PyTorch for tensor operations and neural networks
from torch import nn  # Importing neural network tools from PyTorch
import torch.nn.functional as F  # Importing PyTorch functional API for common operations like activation functions
from torch.optim import Adam  # Importing Adam optimizer for training the model
from torch.utils.data import DataLoader  # Importing DataLoader for batching and shuffling data
import argparse  # Importing argparse for command-line argument parsing
from Model import GAT  # Assuming the GAT model is defined in models.py
import torch  # Importing the core PyTorch library for deep learning operations
import sys
#from DataProcessing.Converting_Dataset_to_DataLoader import load_data_loader
sys.path.append('C:/Users/galrt/PycharmProjects/SolarMLProject/DataProcessing')


#################################
### TRAIN AND TEST FUNCTIONS  ###
#################################

def train_iter(epoch, model, optimizer, criterion, data_loader, print_every=10):
    # start_t = time.time()  # Start tracking time for this iteration
    model.train()  # Set the model to training mode (affects dropout layers, etc.)
    total_loss = 0  # Initialize total loss accumulator for the epoch
    for batch in data_loader:  # Iterate through each batch in the training DataLoader
        optimizer.zero_grad()  # Zero out the gradients from the previous iteration

        # Assuming the batch is structured as (input_tensor, edge_index, edge_attr, target)
        input_tensor, edge_index, edge_attr, target = batch

        # Ensure that input_tensor, edge_index, edge_attr, and target are moved to the correct device (CPU/GPU)
        input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

        # Forward pass through the model
        output = model(input_tensor, edge_index, edge_attr)  # The model takes features, edge_index, and edge_attr
        loss = criterion(output.squeeze(), target)  # Calculate the loss (squeeze to remove extra dimension)

        loss.backward()  # Perform backpropagation to calculate gradients
        optimizer.step()  # Update model parameters based on gradients

        total_loss += loss.item()  # Add the current loss to the total loss accumulator

    avg_loss = total_loss / len(data_loader)  # Calculate the average loss for the epoch

    if epoch % print_every == 0:  # If the epoch number is divisible by print_every, print the loss
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss: {avg_loss:.4f}')

    return avg_loss  # Return the average loss for this epoch

def test(model, criterion, data_loader):
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    total_loss = 0  # Initialize total loss accumulator for the test set
    total_samples = 0  # Initialize the counter for total samples in the test set
    with torch.no_grad():  # Disable gradient calculation (faster inference)
        for batch in data_loader:  # Iterate through each batch in the test DataLoader
            input_tensor, edge_index, edge_attr, target = batch  # Get the batch data (features, edge_index, edge_attr, GHI targets)

            # Ensure that input_tensor, edge_index, edge_attr, and target are moved to the correct device (CPU/GPU)
            input_tensor, edge_index, edge_attr, target = input_tensor.to(device), edge_index.to(device), edge_attr.to(device), target.to(device)

            output = model(input_tensor, edge_index, edge_attr)  # Forward pass through the model
            loss = criterion(output.squeeze(), target)  # Calculate the loss

            total_loss += loss.item() * len(target)  # Accumulate the loss for the current batch
            total_samples += len(target)  # Update the number of samples processed

    avg_loss = total_loss / total_samples  # Calculate the average loss across all test samples
    return avg_loss  # Return the average test loss - helps summarize the model's performance across all batches in the current epoch
    

#Defines hyperparameters like epochs, learning rate, dropout rate, batch size, etc.
# args is the parsed set of command-line arguments used to configure the training.

#################################
############ MAIN  ##############
#################################

if __name__ == '__main__':  # Ensure the code runs only when executed as a script
    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network for Regression')
    # Creating an argument parser for the command line interface (CLI)
    
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 300)')
    # Define the number of epochs for training (recommended: 300)
    
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
    # Define the learning rate for the optimizer (start with 0.005, it's a good default for most cases)
    
    parser.add_argument('--l2', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    # Define the weight decay (L2 regularization) for the optimizer to prevent overfitting
    
    parser.add_argument('--dropout-p', type=float, default=0.6, help='dropout probability (default: 0.6)')
    # Define the dropout probability (recommended: 0.6, this is often good to prevent overfitting in deep learning models)
    
    parser.add_argument('--hidden-dim', type=int, default=64, help='dimension of the hidden representation (default: 64)')
    # Define the size of the hidden layers (64 is a common starting point for this type of model)
    
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads (default: 8)')
    # Define the number of attention heads in the GAT model (8 is typically a good choice for most applications)
    
    parser.add_argument('--concat-heads', action='store_true', default=False, help='whether to concatenate attention heads')
    # Whether to concatenate the results of multiple attention heads (leave as False for regression tasks)
    
    parser.add_argument('--val-every', type=int, default=20, help='epochs to wait for print training and validation evaluation')
    # Define how often to print training and validation results (every 20 epochs is a reasonable interval)
    
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    # Define the batch size used for training (64 is a common batch size and a good starting point)
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # Option to disable CUDA (GPU) training, useful for running the model on CPU if GPU is not available
    
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    # Option to quickly test a single epoch, useful for debugging or testing the setup
    
    parser.add_argument('--seed', type=int, default=13, metavar='S', help='random seed (default: 13)')
    # Set the random seed for reproducibility (13 is used as the default seed, can be changed if needed)
    
    args = parser.parse_args()  # Parse the arguments passed through the command line

    # This line of code is used to set the "random seed" for PyTorch. By setting a random seed, you ensure that the random
    # operations (like weight initialization, data shuffling, etc.) inside PyTorch will produce the same results every time
    # you run your code.
    # Example: In most neural networks, the weights are initialized randomly at the beginning of training.
    # Without a random seed, each run would start with different weights, leading to different training results.
    torch.manual_seed(args.seed)
    
    #This checks if the code should run on the GPU (CUDA-enabled device) or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using {device} device')

    # Load data using the load_data function
    start_time = time.time()
    load_data = load_data_loader("C:/Users/galrt/Desktop/data/pkl", batch_size=32)
    print(load_data)
    end_time_1 = time.time()
    print(f"Program finished running data processing in {end_time_1 - start_time:.2f} seconds.")

    # load_data('train') loads the training data.
	# batch_size=args.batch_size sets the number of samples per batch.
	# shuffle=True shuffles the data before splitting it into batches.
    data_loader_train = DataLoader(load_data('train'), batch_size=args.batch_size, shuffle=True)  # Training DataLoader - This loads the training dataset and prepares it for batching (without shuffling).
    data_loader_val = DataLoader(load_data('val'), batch_size=args.batch_size, shuffle=False)  # Validation DataLoader - This loads the validation dataset and prepares it for batching (without shuffling).
    data_loader_test = DataLoader(load_data('test'), batch_size=args.batch_size, shuffle=False)  # Test DataLoader -  This loads the test dataset and prepares it for batching (without shuffling).

    end_time_2 = time.time()
    print(f"Program finished running data processing 2 in {end_time_2 - start_time:.2f} seconds.")

    # Create the model
    Gat_net = GAT(
        in_features=input_tensor.shape[1],  # Number of input features per node  
        n_hidden=args.hidden_dim,  # Output size of the first Graph Attention Layer
        n_heads=args.num_heads,  # Number of attention heads in the first Graph Attention Layer
        num_classes=1,  # Only one output: GHI (regression task)
        concat=args.concat_heads,  # Whether to concatenate attention heads
        dropout=args.dropout_p,  # Dropout rate
        leaky_relu_slope=0.2  # Alpha (slope) of the leaky relu activation
    ).to(device) # This moves the model to the appropriate device (GPU or CPU).

    # Configure optimizer and loss function for regression
    # gat_net.parameters() passes the model's parameters to the optimizer.
    # lr=args.lr sets the learning rate.
	# weight_decay=args.l2 applies L2 regularization (weight decay) to prevent overfitting
    optimizer = Adam(Gat_net.parameters(), lr=args.lr, weight_decay=args.l2)  # Adam optimizer with learning rate and weight decay
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks - This sets up the loss function, which is Mean Squared Error (MSE), appropriate for regression tasks.

    print("starting training and validation")

    # Train and evaluate the model
    for epoch in range(args.epochs):  # Loop over the specified number of epochs
        train_loss = train_iter(epoch + 1, Gat_net, optimizer, criterion, data_loader_train, args.val_every)  # Train for one epoch
        
        #This condition checks if it's time to evaluate the model on a validation set. 
        # Specifically, it ensures that the model isn't evaluated on the validation set too often, 
        # which might be inefficient, but also frequently enough to monitor progress and prevent overfitting
        if epoch % args.val_every == 0:  # If the epoch number is divisible by val_every, run validation
            val_loss = test(Gat_net, criterion, data_loader_val)  # Test the model on the validation set and computes the loss
            print(f'Validation loss: {val_loss:.4f}')  # Print validation loss

        if args.dry_run:  # If dry_run is enabled, break after one epoch
            break

    end_time_3 = time.time()
    print(f"Program finished training and validation in {end_time_3 - start_time:.2f} seconds.")

    # Final test
    test_loss = test(Gat_net, criterion, data_loader_test)  # Evaluate the model on the test set

    print(f'Test set results: loss {test_loss:.4f}')  # Print the test loss
    end_time_4 = time.time()
    print(f"Program finished testingin {end_time_4 - start_time:.2f} seconds.")
