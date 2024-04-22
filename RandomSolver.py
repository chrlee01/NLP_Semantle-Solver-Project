import json
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import random
import torch
import numpy as np
from tqdm import tqdm 
from annoy import AnnoyIndex

class RandomHelper():
    def __init__(self, word_file_path='words.json', model_name='word2vec-google-news-300'):
        #loads word list
        with open(word_file_path, 'r') as fp:
            self.word_list = json.load(fp)['words']
        self.model = api.load(model_name)
        self.verbose = False

    def _get_similarity_score(self, guessed_word, target_word):
        if guessed_word == target_word:
            return 100
        if guessed_word not in self.model:
            return 0
        guessed_vector = self.model[guessed_word].reshape(1, -1)
        target_vector = self.model[target_word].reshape(1, -1)
        similarity = (cosine_similarity(target_vector, guessed_vector)[0][0] * 100)
        return similarity
    def _get_guess(self):
        return np.random.choice(self.word_list)
    
    def simulate_games(self, num_games=20, max_guesses=50):
        print("Simulating games...")
        total_guesses = 0
        total_average_similarity = 0  # To track the average similarity per game
        progress_bar = tqdm(range(num_games), desc=f"Simulating {num_games} games")
        
        for _ in progress_bar:
            target_word = np.random.choice(self.word_list)
            similarities = []  # List to keep track of similarity scores in the current game
            
            for i in range(max_guesses):
                total_guesses += 1
                guessed_word = np.random.choice(self.word_list)
                similarity = self._get_similarity_score(guessed_word, target_word)
                similarities.append(similarity)  # Add the similarity score to the list
                
                if guessed_word == target_word:
                    break
            
            # Calculate average similarity for the current game
            average_similarity = np.mean(similarities)
            total_average_similarity += average_similarity
            progress_bar.set_postfix({'Avg Sim': f'{average_similarity:.2f}'})  # Display average similarity for the current game

        average_guesses = total_guesses / num_games
        overall_average_similarity = total_average_similarity / num_games  # Calculate the overall average similarity across all games
        print(f'Average number of guesses per game: {average_guesses}')
        print(f'Overall average similarity per game: {overall_average_similarity:.2f}')



helper = RandomHelper()
helper.simulate_games(num_games=100, max_guesses=50)
        

    