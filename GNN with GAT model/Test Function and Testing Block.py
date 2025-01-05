import torch  # Importing PyTorch for tensor operations and neural networks

#######################
### TEST FUNCTION  ###
#######################

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
    
 # Final test
    test_loss = test(gat_net, criterion, data_loader_test)  # Evaluate the model on the test set
    print(f'Test set results: loss {test_loss:.4f}')  # Print the test loss
