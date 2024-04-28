from dqn_agent import Agent
from SemantleGameEnv import SemantleEnv
import json
import numpy as np
import matplotlib.pyplot as plt
import os  


def plot_scores(similarity_scores):
    """
    Plots the average similarity score over time
    """
    plt.figure(figsize=(10, 5))
    plt.plot(similarity_scores, label='Average Similarity Score')
    plt.title('Average Similarity Scores Per Game')
    plt.xlabel('Game Number')
    plt.ylabel('Average Similarity Score')
    plt.legend()
    plt.show()


# Load the list of words for the game
with open('../words.json', 'r') as fp:
    word_list = json.load(fp)['words']


if __name__ == '__main__':
    """
    Main function to run the DQN solver
    """
    print("start")
    # Initialize the Semantle environment with the word list
    env = SemantleEnv(word_list=word_list, 
                      history_length=5, 
                      max_guesses=50,
                      correct_guess_bonus=200,
                      incorrect_guess_penalty=0)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(gamma=0.99, 
                   epsilon=1.0,
                   batch_size=256,
                   n_actions=action_size,
                   eps_end=0.01,
                   input_dims=state_size,
                   lr=0.001)

    model_path = 'dqn_model.pth'
    if os.path.exists(model_path):
        print("Loading existing model...")
        agent.load_model(model_path)
    else:
        print("No existing model found, starting fresh...")

    # Number of games to play
    n_games = 1000
    # Save model every so many games
    save_interval = 1000

    # Keep track of similarity scores per game
    similarity_scores = []
    # Keep track of average similarity scores over the last 100 games
    averages = []
    for i in range(n_games):
        
        if (i % 100 == 0):
            print(" ")
            print("Game " + str(i))
            env.verbose = True
            print(" ")
        else:
            env.verbose = False
            
        done = False
        score = 0
        
        observation = env.reset()
        game_scores = []
        while not done:
            #loop for each game, runs until game is over
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
        local_average = np.mean(similarity_scores[-100:])
        averages.append(local_average)
        print(f'episode {i}, score {score:.2f}, game average similarity score {np.mean(game_scores):.2f}, recent average score {local_average:.2f}')
    print("end")
    plot_scores(averages)


