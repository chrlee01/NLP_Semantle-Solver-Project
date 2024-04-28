import datetime
import json
import time

import gensim.downloader as api
import pytz
from sklearn.metrics.pairwise import cosine_similarity

import constants


class SemantleClient:

    def __init__(self, model=None, verbose=False):
        self.verbose = verbose
        self.loaded_model = None
        self._load_word2vec_model(model=model)

    def _load_word2vec_model(self, model):
        time_first = time.perf_counter()
        if model is None:
            model = constants.WORD2VEC_MODEL
        if self.verbose:
            print('Loading model...')
        self.loaded_model = api.load(model)
        elapsed_time = time.perf_counter() - time_first
        if self.verbose:
            print(f"Model loaded in {elapsed_time} seconds.")

    def initialize_game(self, day_number=None):
        return SemantleSession(day_number, self)


class SemantleSession:

    def __init__(self, day_number, client):
        self.client = client
        self.day = day_number
        if self.day is None:
            self._get_current_day_number()
        self.target = None
        self.target_vector_comparison = None
        self._get_target()
        self.url = None
        self._generate_url()

    def _get_current_day_number(self):
        start_date = datetime.date(*constants.SEMANTLE_START_DATE)
        today = datetime.datetime.now(pytz.utc).date()
        curr_semantle_number = (today - start_date).days
        if self.client.verbose:
            print(f'Playing semantle #{curr_semantle_number}')
        self.day = curr_semantle_number

    def _get_target(self):
        with open(constants.WORDS_FILE, 'r') as fp:
            words = json.load(fp)['words']
        self.target = words[self.day]
        target_vector = self.client.loaded_model[self.target]
        self.target_vector_comparison = target_vector.reshape(1, -1)

    def _generate_url(self):
        self.url = f'{constants.SEMANTLE_BASE_URL}/{self.target}'

    def check_guess(self, guess):
        if guess == self.target:
            return {'guess': guess, 'correct': True, 'similarity': 100, 'invalid': False}
        elif guess not in self.client.loaded_model:
            return {'guess': guess, 'correct': False, 'similarity': None, 'invalid': True}
        else:
            calculated_vector_comparison = self.client.loaded_model[guess].reshape(1, -1)

            cosine_similarity_ = cosine_similarity(self.target_vector_comparison, calculated_vector_comparison)[0][0]
            similarity = cosine_similarity_ * 100
            return {'guess': guess, 'correct': False, 'similarity': similarity, 'invalid': False}

    def play(self):
        history = []
        while True:
            guess = input('Enter a guess: ')
            result = self._check_guess(guess)
            history.append(result)
            if result['correct']:
                if self.client.verbose:
                    print('Correct!')
                    break
            elif result['invalid']:
                if self.client.verbose:
                    print('Invalid Guess.')
                    continue
            else:
                if self.client.verbose:
                    similarity = result['similarity']
                    print(f'Incorrect. Similarity: {similarity}.')
        return history


if __name__ == '__main__':
    client = SemantleClient(verbose=True)
    game = client.initialize_game()
    game.play()
