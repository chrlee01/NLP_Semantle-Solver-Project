import torch.nn as nn
import torch.nn.functional as F

class SemantleSolver(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2):
        """
        Initialize the SemantleSolver model.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            hidden_dim1 (int): The number of neurons in the first hidden layer.
            hidden_dim2 (int): The number of neurons in the second hidden layer.

        Initializes the model with three fully connected layers:
        - The first layer has `embedding_dim + 1` input neurons, `hidden_dim1` output neurons, and uses ReLU activation.
        - The second layer has `hidden_dim1` input neurons, `hidden_dim2` output neurons, and uses ReLU activation.
        - The third layer has `hidden_dim2` input neurons, `embedding_dim` output neurons, and uses the identity activation function.

        Returns:
            None
        """
        super(SemantleSolver, self).__init__()
        self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim1)  # +1 for the similarity score
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, embedding_dim)  # Output the size of the embedding

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor of the network after passing through the ReLU activation functions and the final linear layer.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Use the correct output layer
        return x