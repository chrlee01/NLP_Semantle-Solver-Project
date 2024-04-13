from dqn_agent import Agent
from SemantleGameEnv import SemantleEnv  # Assuming the environment class is defined here
import json
import numpy as np

# Load the list of words for the game
with open('words.json', 'r') as fp:
    word_list = json.load(fp)['words']

if __name__ == '__main__':
    print("start")
    # Initialize the Semantle environment with the word list
    env = SemantleEnv(word_list=word_list, history_length=10, max_guesses=100, correct_guess_bonus=500, incorrect_guess_penalty=-100)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=128, n_actions=action_size, eps_end=0.01, input_dims=[state_size], lr=0.003)
    
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(f'episode {i}, score {score:.2f}, average score {avg_score:.2f}')
    print("end")
