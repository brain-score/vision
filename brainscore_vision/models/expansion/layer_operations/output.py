from torch import nn
import torch

class Output(nn.Module):
    """
    A module that reshapes input tensors into a two-dimensional format suitable for output layers.
    This class simplifies the tensor to be batch_size by features, collapsing all other dimensions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that reshapes the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be reshaped.

        Returns:
            torch.Tensor: The reshaped tensor, with shape (N, -1) where N is the batch size.
        """
        N = x.shape[0]  # Get the batch size from the input tensor
        return x.reshape(N, -1)
