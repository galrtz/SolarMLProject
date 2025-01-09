import torch  # Importing the core PyTorch library for deep learning operations
from torch import nn  # Importing PyTorch's neural network module to define and manage neural networks
import torch.nn.functional as F  # Importing PyTorch's functional API for commonly used functions like activations (e.g., relu, softmax).
from torch_geometric.nn import GATConv

# This defines a custom class GAT that extends the base PyTorch class nn.Module,
# meaning it's a neural network module and can be used for training and inference:
class GAT(nn.Module):

    # __init__: The constructor method used to initialize the GAT model.
    # Parameters:
    # in_features: Number of input features for each node in the graph.
    # n_hidden: Number of output features per node after the first Graph Attention Layer (GAT).
    # n_heads: Number of attention heads in the first GAT layer.
    # concat: Whether to concatenate the attention heads' outputs (default is False).
    # dropout: Dropout rate for regularization (default is 0.4).
    # leaky_relu_slope: Slope of the Leaky ReLU activation (default is 0.2).
    # num_classes: After all attention layers - num_classes should correspond to 1,
    # as you're predicting a single continuous value (the GHI at a certain time).

    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4, leaky_relu_slope=0.2):
        # Initializes the parent class (nn.Module) to ensure that the model works correctly with PyTorch's internal
        # functionality like automatic differentiation, model saving, and others.
        super(GAT, self).__init__()

        # Define First Graph Attention Layer
        self.gat1 = GATConv(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, negative_slope=leaky_relu_slope
        )

        # Define Second Graph Attention Layer
        self.gat2 = GATConv(
            in_features=n_hidden, out_features=1, n_heads=1,
            concat=False, dropout=dropout, negative_slope=leaky_relu_slope
        )

    # The forward method defines how the model processes input data and computes the output.
    def forward(self, input_tensor: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Performs a forward pass through the GAT.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            edge_index (torch.Tensor): Edge index tensor representing graph structure.
            edge_attr (torch.Tensor): Edge attribute tensor (optional, depending on your use case).

        Returns:
            torch.Tensor: Continuous output tensor after the forward pass (predicted GHI values).
        """
        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, edge_index, edge_attr)

        # Apply the ELU (Exponential Linear Unit) activation function to the output of the first GAT layer.
        x = F.elu(x)  # Apply ELU activation function

        # Apply the second Graph Attention layer
        x = self.gat2(x, edge_index, edge_attr)

        # Return the final output from the second GAT layer.
        return x

    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv

    class GAT(nn.Module):
        def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=False, dropout=0.4,
                     leaky_relu_slope=0.2):
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
