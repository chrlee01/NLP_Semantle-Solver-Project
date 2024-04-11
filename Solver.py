from semantle_reverse_engineer import SemantleClient
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQSolver:
    def __init__(self, state_size, action_size, verbose=False):
        self.verbose = verbose
        self.client = SemantleClient(verbose=verbose)
        self.game = self.client.initialize_game()
        self.possible_answers = self._load_5000_common_words()

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _predict_guess(self):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    # play the game while learning
    def play(self):
        history = []
        while True:
            guess = self._predict_guess()
            result = self.game.check_guess(guess)
            history.append(result)

            # TODO: feed result back into model somehow here
            if result['correct']:
                if self.client.verbose:
                    print('Correct!')
                    break
            elif result['invalid']:
                if self.client.verbose:
                    print('Invalid Guess.')
                    continue
            else:
                if self.client.verbose:
                    similarity = result['similarity']
                    print(f'Incorrect. Similarity: {similarity}.')

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return history