import pickle
import torch

def save_model(model, filename):
    """
    Save the PyTorch model to a file.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - filename (str): The name of the file to save the model to.
    """
    torch.save(model.state_dict(), filename)  # Save only the model's state_dict (parameters)

def load_model(model_class, filename, device='cpu', **model_args):
    """
    Load the PyTorch model from a file.

    Parameters:
    - model_class (class): The class of the model to be loaded.
    - filename (str): The name of the file to load the model from.
    - device (str): The device to load the model onto ('cpu' or 'cuda').
    - model_args (dict): Any arguments required to initialize the model.

    Returns:
    - torch.nn.Module: The loaded model.
    """
    model = model_class(**model_args)  # Instantiate the model
    model.load_state_dict(torch.load(filename, map_location=device))  # Load the state_dict
    model.to(device)  # Move the model to the appropriate device
    return model

# Example usage:
# save_model(Gat_net, 'gat_model_weights.pth')
# loaded_model = load_model(GAT, 'gat_model_weights.pth', device='cuda', in_features=..., n_hidden=..., n_heads=..., num_classes=..., concat=..., dropout=..., leaky_relu_slope=...)
