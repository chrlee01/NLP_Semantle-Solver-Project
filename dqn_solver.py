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
    env = SemantleEnv(word_list=word_list, history_length=5, max_guesses=100, correct_guess_bonus=400, incorrect_guess_penalty=-500)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=action_size, eps_end=0.01, input_dims=state_size, lr=0.001)

    
    n_games = 30000
    save_interval = 1000  # Save the model every 1000 games
    
    similarity_scores = []
    for i in range(n_games):
        
        if (i % 100 == 0):
            print(" ")
            print("Game " + str(i))
            env.verbose = True  # Fix to access the instance variable correctly
            print(" ")
        else:
            env.verbose = False
            
        done = False
        score = 0
        
        observation = env.reset()
        game_scores = []
        while not done:
            
            action = agent.choose_action(observation)
            observation_, reward, done, info, similarity_score = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            game_scores.append(similarity_score)
            
        similarity_scores.append(np.mean(game_scores))
            # Save model periodically and at the end of training
        if (i + 1) % save_interval == 0 or (i + 1) == n_games:
            agent.save_model()
        print(f'episode {i}, score {score:.2f}, game average similarity score {np.mean(game_scores):.2f}, recent average score {np.mean(similarity_scores[-100:]):.2f}')
    print("end")

    plot_scores(similarity_scores)

