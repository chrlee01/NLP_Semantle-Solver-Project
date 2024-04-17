from Helper import Helper
from SemantleSolver import SemantleSolver
from torch import nn
import torch
import numpy as np



#parameters to tweak
EMBEDDING_DIM = 300  #using word2vec
HIDDEN_DIM = 512
HIDDEN_DIM2 = 512
NUMEXAMPLES = 1000
EPOCHS = 30
NUM_GAMES_TO_SIMULATE = 10
MAX_GUESSES_PER_GAME = 50
LEARNING_RATE = 0.001


if __name__ == "__main__":
    
    # Check if CUDA is available and set the default device accordingly
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
        print("GPU is available. Set default dtype to Float32 and device to CUDA.")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cpu')
        print("GPU is not available. Using CPU and set default dtype to Float32.")
        
    helper = Helper()  # Assuming the class from Helper.py is available
    inputs, outputs = helper.generate_training_examples(NUMEXAMPLES)
    model = SemantleSolver(EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM2)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    helper.train_model(model, criterion, optimizer, inputs, outputs, EPOCHS)
    
    # Simulate a few games to evaluate the model
    helper.simulate_games(model, num_games=NUM_GAMES_TO_SIMULATE, max_guesses=MAX_GUESSES_PER_GAME)