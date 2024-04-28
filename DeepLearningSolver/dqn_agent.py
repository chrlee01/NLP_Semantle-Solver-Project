import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    """Initialize parameters and optimizer for the Deep Q Network."""
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        """Initialize the Deep Q Network.

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer.
        input_dims : int
            The number of input dimensions (size of observation).
        fc1_dims : int
            The number of nodes in the first fully connected layer.
        fc2_dims : int
            The number of nodes in the second fully connected layer.
        n_actions : int
            The number of possible actions.
        """
        super(DeepQNetwork, self).__init__()
        # Ensure input_dims is an integer
        input_dims = input_dims if isinstance(input_dims, int) else input_dims[0]

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(T.cuda.is_available())
        self.to(self.device)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    """Deep Q-Network Agent for the Solver environment."""
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        """
        Initializes the Deep Q-Network Agent with the given parameters.

        Parameters:
            gamma (float): discount factor
            epsilon (float): epsilon-greedy parameter
            lr (float): learning rate
            input_dims (int): dimensions of the input
            batch_size (int): batch size
            n_actions (int): number of possible actions
            max_mem_size (int, optional): maximum memory size. Defaults to 100000.
            eps_end (float, optional): minimum epsilon. Defaults to 0.01.
            eps_dec (float, optional): epsilon decay rate. Defaults to 5e-4.

        Returns:
            None
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        
        self.Q_eval = DeepQNetwork(lr, input_dims, 512, 512, n_actions)
        
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, state_, done):
        """Store a transition into the memory."""
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self, observation):
        """Choose an action based on an epsilon-greedy policy or a random action."""
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        """Update the Q-network using the replay memory."""
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]
        
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
    def save_model(self, filename="dqn_model.pth"):
        """Save the Q-network to a file."""
        T.save({
            'model_state_dict': self.Q_eval.state_dict(),
            'optimizer_state_dict': self.Q_eval.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load_model(self, filename="dqn_model.pth"):
        """Load the Q-network from a file."""
        checkpoint = T.load(filename, map_location=self.Q_eval.device)
        self.Q_eval.load_state_dict(checkpoint['model_state_dict'])
        self.Q_eval.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
