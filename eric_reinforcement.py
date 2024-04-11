from semantle_reverse_engineer import SemantleClient
import json

class DeepQSolver:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.client = SemantleClient(verbose=verbose)
        self.possible_answers = self._load_5000_common_words()

    def _load_10000_possible_guesses(self) -> list:
        with open('words_dictionary.json', 'r') as fp:
            words = json.load(fp)['words']
        return words
    
    def _load_5000_common_words(self) -> list:
        with open('words.json', 'r') as fp:
            words = json.load(fp)['words']
        return words
    
    