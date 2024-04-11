import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.tensor(reward)
            action = torch.tensor(action)
            done = torch.tensor(done)
            
            target = reward + (1 - done) * self.gamma * torch.max(self.model(next_state))
            current_q = self.model(state).gather(1, action.unsqueeze(1))
            
            loss = nn.MSELoss()(current_q, target.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    # Set up your environment
    env = gym.make('CartPole-v1')  # Example for a Gym environment, replace with your environment if different

    # Define parameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    batch_size = 32
    episodes = 1000  # Number of episodes for training

    # Create DQN agent
    agent = DQNAgent(state_size, action_size)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Interact with the environment
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the agent
            agent.replay(batch_size)

        # Update exploration rate
        agent.epsilon *= agent.epsilon_decay
        agent.epsilon = max(agent.epsilon_min, agent.epsilon)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")