from dqn_agent import Agent
from SemantleGameEnv import SemantleEnv  # Assuming the environment class is defined here
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_scores(similarity_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(similarity_scores, label='Average Similarity Score')
    plt.title('Average Similarity Scores Per Game')
    plt.xlabel('Game Number')
    plt.ylabel('Average Similarity Score')
    plt.legend()
    plt.show()

# Load the list of words for the game
with open('words.json', 'r') as fp:
    word_list = json.load(fp)['words']

if __name__ == '__main__':
    print("start")
    # Initialize the Semantle environment with the word list
    env = SemantleEnv(word_list=word_list, history_length=100, max_guesses=100, correct_guess_bonus=400, incorrect_guess_penalty=-50)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(gamma=0.95, epsilon=1.0, batch_size=256, n_actions=action_size, eps_end=0.05, input_dims=[state_size], lr=0.002)
    
    scores, eps_history = [], []
    n_games = 1000
    similarity_scores = []
    for i in range(n_games):
        
        if (i % 100 == 0):
            print(" ")
            print("Game " + str(i))
            SemantleEnv.verbose = True
            print("Average Similarity: " + str(np.mean(env.state[1::2])))
            print(" ")
            #prints average similarity score of this game
        else:
            SemantleEnv.verbose = False
            
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
        avg_score = np.mean(env.state[1::2])
        similarity_scores.append(avg_score)
        print(f'episode {i}, score {score:.2f}, average similarity score {np.mean(similarity_scores)}')
    print("end")

    plot_scores(similarity_scores)


