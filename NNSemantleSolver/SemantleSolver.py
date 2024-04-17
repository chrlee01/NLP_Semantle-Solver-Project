import torch.nn as nn
import torch.nn.functional as F

class SemantleSolver(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2):
        super(SemantleSolver, self).__init__()
        self.fc1 = nn.Linear(embedding_dim + 1, hidden_dim1)  # +1 for the similarity score
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, embedding_dim)  # Output the size of the embedding

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Use the correct output layer
        return x