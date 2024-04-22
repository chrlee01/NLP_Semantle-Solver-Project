from Helper import Helper
from SemantleSolver import SemantleSolver
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import os



#parameters to tweak
EMBEDDING_DIM = 300  #using word2vec
HIDDEN_DIM = 512
HIDDEN_DIM2 = 512
NUMEXAMPLES = 15000
EPOCHS = 50
NUM_GAMES_TO_SIMULATE = 10
MAX_GUESSES_PER_GAME = 10000
LEARNING_RATE = 0.0005
MODEL_SAVE_PATH = 'semantle_model_2.pth'
MODEL_LOAD_PATH = 'semantle_model.pth'

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Average Loss per Example During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss per Training Example')
    plt.show()


if __name__ == "__main__":
    
    # Check if CUDA is available and set the default device accordingly
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
        print("GPU is available. Set default device to CUDA.")
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cpu')
        print("GPU is not available. Using CPU")

        
    helper = Helper()  # Assuming the class from Helper.py is available
    inputs, outputs = helper.generate_training_examples(NUMEXAMPLES)
    model = SemantleSolver(EMBEDDING_DIM, HIDDEN_DIM, HIDDEN_DIM2)
    
    if os.path.exists(MODEL_LOAD_PATH):
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))
        print("Loaded existing model.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    # Train the model
    losses = helper.train_model(model, criterion, optimizer, inputs, outputs, EPOCHS)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Saved model after training.")
    #plot_losses(losses)
    
    
    # Simulate a few games to evaluate the model
    helper.simulate_games(model, num_games=NUM_GAMES_TO_SIMULATE, max_guesses=MAX_GUESSES_PER_GAME)
    
    
