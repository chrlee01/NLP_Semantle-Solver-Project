from semantle_reverse_engineer import SemantleClient
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class EricSolver:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.client = SemantleClient(verbose=verbose)
        self.possible_answers = self._load_5000_common_words()

    def _load_possible_guesses(self) -> list:
        with open('words_dictionary.json', 'r') as fp:
            words = json.load(fp)['words']
        return words
    
    def _load_5000_common_words(self) -> list:
        with open('words.json', 'r') as fp:
            words = json.load(fp)['words']
        return words
    
    def _plot_similarities(self, history):
        similarities = [result['similarity'] for result in history]
        plt.plot(similarities)
        plt.xlabel('Guesses')
        plt.ylabel('Similarity')
        plt.show()
    
    def solve(self, day=0):
        self.session = self.client.initialize_game(day_number=day)
        self.possible_guesses = self.possible_answers.copy()
        # self.possible_guesses = self._load_possible_guesses()
        guess = random.choice(self.possible_answers)
        print("Starting guess:", guess)
        history = []

        while self.possible_guesses:
            if guess not in self.possible_guesses:
                guess = random.choice(self.possible_guesses)
            self.possible_guesses.remove(guess)
            result = self.session._check_guess(guess)
            history.append(result)

            if self.verbose:
                print(f'Guess: {guess}')
            if result['correct']:
                # if self.verbose:
                print(f'Correct! Took {len(history)} guesses.')
                break
            elif result['invalid']:
                if self.verbose:
                    print('Invalid Guess.')
                    continue
            else:
                if self.verbose:
                    similarity = result['similarity']
                    print(f'Incorrect. Similarity: {similarity}.')
            guess = self._guess(history)
        print('Word was:', self.session.target)
        # self._plot_similarities(history)
        return history

    def _guess(self, history):
        model = self.client.loaded_model

        word_data = history[-1]
        curr_guess = word_data['guess']
        curr_similarity = word_data['similarity']

        for guess in self.possible_guesses:
            if guess not in model:
                self.possible_guesses.remove(guess)
                continue
            calculated_vector_comparison = model[guess].reshape(1, -1)
            target_vector_comparison = model[curr_guess].reshape(1, -1)
            cosine_similarity_ = cosine_similarity(target_vector_comparison, calculated_vector_comparison)[0][0]
            similarity = cosine_similarity_ * 100

            if similarity != curr_similarity:
                self.possible_guesses.remove(guess)
            else:
                return guess

        if len(self.possible_guesses) == 0:
            return ""
                
        return random.choice(self.possible_guesses)

num_guesses_history = []

if __name__ == '__main__':
    solver = EricSolver(verbose=False)
    for i in range(500):
        print(f'Game {i}')
        history = solver.solve(i)
        num_guesses_history.append(len(history))

    plt.plot(num_guesses_history)
    plt.xlabel('Games')
    plt.ylabel('Number of Guesses')
    plt.show()