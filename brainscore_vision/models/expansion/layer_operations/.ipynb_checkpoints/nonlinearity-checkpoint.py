import torch
from torch import nn

OPERATION_TYPES = ['zscore', 'leaky_relu', 'relu', 'gelu', 'abs', 'elu', 'none']

class NonLinearity(nn.Module):
    """
    A neural network module to apply various non-linear operations to input tensors.

    Attributes:
        operation (str): The type of non-linear operation to apply (e.g., 'zscore', 'relu').
        operation_type (list): A list of supported non-linear operations.
    """
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation
        self.operation_type = OPERATION_TYPES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the non-linearity module that applies the specified operation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to which the non-linearity will be applied.

        Returns:
            torch.Tensor: The tensor after the non-linearity has been applied.

        Raises:
            AssertionError: If the operation specified is not supported.
        """
        assert self.operation in self.operation_type, f'Invalid operation type, choose one of {self.operation_type}'

        match self.operation:
            case 'zscore':
                std = x.std(dim=1, keepdims=True)
                mean = x.mean(dim=1, keepdims=True)
                return (x - mean) / std

            case 'elu':
                return nn.ELU(alpha=1.0)(x)
        
            case 'leaky_relu': 
                return nn.LeakyReLU()(x)

            case 'relu': 
                return nn.ReLU()(x)
            
            case 'gelu': 
                return nn.GELU()(x)

            case 'abs': 
                return x.abs()
            
            case 'none': 
                return x
