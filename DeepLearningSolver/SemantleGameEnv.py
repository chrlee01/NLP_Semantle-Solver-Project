import gym
from gym import spaces
import numpy as np
import random
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import math

class SemantleEnv(gym.Env):
    def __init__(self, word_list, history_length=5, max_guesses=100, correct_guess_bonus=1000, incorrect_guess_penalty=-500, model_name='word2vec-google-news-300'):
        super(SemantleEnv, self).__init__()
        self.word_list = word_list
        self.history_length = history_length
        self.max_guesses = max_guesses
        self.correct_guess_bonus = correct_guess_bonus
        self.model = api.load(model_name)  # Load GloVe model
        self.incorrect_guess_penalty = incorrect_guess_penalty
        self.action_space = spaces.Discrete(len(self.word_list))
        
        # Determine size of each entry in the state (300 for the embedding + 1 for the similarity score)
        embedding_size = 300
        entry_size = embedding_size + 1  # Plus one for the similarity score
        self.observation_space = spaces.Box(low=-1, high=1, shape=(history_length * entry_size,), dtype=np.float32)
        self.state = np.zeros(self.history_length * entry_size)  # Initialize state array
        self.verbose = False
        self.reset()

    def step(self, action):
        """
        Perform a step in the environment based on the given action.
        
        Parameters:
            action: An integer representing the index of the action to take.
        
        Returns:
            Tuple containing the updated state, the reward earned from the step, 
            a boolean indicating if the episode is done, an empty dictionary, and the similarity score.
        """
        guessed_word = self.word_list[action]
        similarity_score = self._get_similarity_score(guessed_word)
        guessed_vector = self.model[guessed_word] if guessed_word in self.model else np.zeros(300)

        # Define the embedding size and the entry size (embedding + 1 for similarity score)
        embedding_size = 300
        entry_size = embedding_size + 1

        # Shift existing data to the right to make room for the new data at the front
        # The total entries in the buffer are history_length, and each entry is of size entry_size
        self.state[entry_size:] = self.state[:-entry_size]  # Shift everything to the right

        # Insert the new embedding and similarity score at the front of the state
        self.state[0:embedding_size] = guessed_vector
        self.state[embedding_size] = similarity_score

        # Logic to handle game progress and check for game end
        
        if(self.current_guess_count > self.max_guesses):
            done = True
            reward = self.incorrect_guess_penalty
        elif(similarity_score == 100):
            print("Correct guess!, Guessed Word : " + self.word_list[action])
            reward = self.correct_guess_bonus
            done = True
        else:
            reward = similarity_score
            done = False
        
        self.current_guess_count += 1
        return self.state, reward, done, {}, similarity_score

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        self.current_guess_count = 0
        self.target_word = random.choice(self.word_list)
        self.target_vector = self.model[self.target_word].reshape(1, -1) if self.target_word in self.model else np.zeros((1, 300))
        # Initialize the state with zeros to accommodate all embeddings and similarity scores
        embedding_size = 300
        entry_size = embedding_size + 1
        self.state = np.zeros(self.history_length * entry_size)
        return self.state

    def _get_similarity_score(self, guessed_word):
        """
        Calculates the similarity score between a guessed word and the target word.

        Parameters:
            guessed_word (str): The word to compare similarity with the target word.

        Returns:
            float: The similarity score between the guessed word and the target word, ranging from 0 to 100.
                   Returns 100 if the guessed word is equal to the target word.
                   Returns 0 if the guessed word is not in the model.
        """
        if guessed_word == self.target_word:
            return 100
        if guessed_word not in self.model:
            return 0
        guessed_vector = self.model[guessed_word].reshape(1, -1)
        similarity = (cosine_similarity(self.target_vector, guessed_vector)[0][0] * 100)
        return similarity

    def render(self, mode='human'):
        pass  # Not implemented for now
    
    def decay_function(self, x):
        """
        Calculates the decay function for a given value x. Can be used to reduce the value of the reward if the guess happens later in the game.

        Parameters:
            x (float): The input value for the decay function.

        Returns:
            float: The result of the decay function.

        The decay function is defined as e^(-x/20).
        """
        return math.exp(-x/20)
