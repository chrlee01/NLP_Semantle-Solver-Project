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
        self.model = api.load(model_name)
        self.incorrect_guess_penalty = incorrect_guess_penalty
        self.action_space = spaces.Discrete(len(self.word_list))
        
        # State will include the last N guesses and their similarity scores
        # Each guess has 2 values: index and score, so we multiply history_length by 2
        self.observation_space = spaces.Box(low=-100, high=100, shape=(history_length*2,), dtype=np.float32)
        
        self.target_word = random.choice(self.word_list)
        self.current_guess_count = 0
        self.state = np.zeros(history_length * 2)
        self.reset()
        

    def step(self, action):
        
        guessed_word = self.word_list[action]
        similarity_score = self._get_similarity_score(guessed_word)
        
        current_index = (self.current_guess_count * 2) % (self.history_length * 2)
        self.state[current_index] = action  # Store the index of the guessed word in the state history
        self.state[current_index + 1] = similarity_score  # Store the similarity score in the state history
        
        self.current_guess_count += 1
        
        if(self.current_guess_count > self.max_guesses):
            done = True
            reward = self.incorrect_guess_penalty
        elif(similarity_score == 100):
            done = True
            reward = self.correct_guess_bonus
        else:
            reward = similarity_score
            done = False
        
        decay_factor = self.decay_function(self.current_guess_count)  # exponentail decay that we can use to limit rewards obtained at later guesses
        
        # Update state and continue
        return self.state, reward, done, {}


    def reset(self):
        self.current_guess_count = 0
        self.target_word = random.choice(self.word_list)
        self.target_vector = self.model[self.target_word].reshape(1, -1)
        # Initialize the state with -1 for indices and 0 for scores
        self.state = np.zeros(self.history_length * 2)
        return self.state

    def _get_similarity_score(self, guessed_word):
        if(guessed_word == self.target_word):
            return 100
        if guessed_word not in self.model:
            return 0  # Or some other handling of out-of-vocabulary words
        guessed_vector = self.model[guessed_word].reshape(1, -1)
        similarity = (cosine_similarity(self.target_vector, guessed_vector)[0][0] * 100)
        return similarity

    def render(self, mode='human'):
        pass  # Not implemented for now
    
    def decay_function(self, x):
        return math.exp(-x/20)
