import json
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import random
import torch
import numpy as np
from tqdm import tqdm 
from annoy import AnnoyIndex

class Helper():
    def __init__(self, word_file_path='words.json', model_name='word2vec-google-news-300'):
        #loads word list
        with open(word_file_path, 'r') as fp:
            self.word_list = json.load(fp)['words']
        self.model = api.load(model_name)
        self.verbose = False
        
        # Annoy index for fast nearest neighbor lookup
        embedding_dim = self.model.vector_size
        self.index = AnnoyIndex(embedding_dim, 'angular')
        
        # Add all items to the index with a progress bar
        print("Building Annoy index...")
        for i, word in tqdm(enumerate(self.model.index_to_key), total=len(self.model.index_to_key), desc="Indexing words"):
            self.index.add_item(i, self.model[word])
        
        print("Building Trees...")
        self.index.build(10)
        print("Annoy index built successfully.")
        
    def _get_similarity_score(self, guessed_word, target_word):
        """
        Calculates the similarity score between two words.

        Parameters:
            guessed_word (str): The word to compare similarity with the target word.
            target_word (str): The word to compare similarity with the guessed word.

        Returns:
            float: The similarity score between the guessed word and the target word, ranging from 0 to 100.
                   Returns 100 if the guessed word is equal to the target word.
                   Returns 0 if the guessed word is not in the model.
        """
        if guessed_word == target_word:
            return 100
        if guessed_word not in self.model:
            return 0
        guessed_vector = self.model[guessed_word].reshape(1, -1)
        target_vector = self.model[target_word].reshape(1, -1)
        similarity = (cosine_similarity(target_vector, guessed_vector)[0][0] * 100)
        return similarity
    def find_closest_word(self, predicted_embedding):
        """
        Finds the closest word to the given predicted embedding.

        Args:
            predicted_embedding (numpy.ndarray): The predicted embedding.

        Returns:
            str: The closest word to the predicted embedding.
        """
        closest_id = self.index.get_nns_by_vector(predicted_embedding, 1)[0]
        return self.model.index_to_key[closest_id]
    
    def generate_training_examples(self, num_examples):
        """
        Generates training examples for a machine learning model.

        Args:
            num_examples (int): The number of training examples to generate.

        Returns:
            tuple: A tuple containing the inputs and outputs as numpy arrays.
        """
        print("Generating training examples...")
        inputs = []
        outputs = []
        for _ in tqdm(range(num_examples), desc="Generating training data"):
            target_word = random.choice(self.word_list)
            target_vector = self.model[target_word].reshape(1, -1)
            guessed_word = random.choice(self.word_list)
            if guessed_word not in self.model:
                continue
            guessed_vector = self.model[guessed_word].reshape(1, -1)
            similarity_score = self._get_similarity_score(guessed_word, target_word)
            input_vector = np.concatenate((guessed_vector.flatten(), [similarity_score]))
            inputs.append(input_vector)
            outputs.append(target_vector.flatten())
        print(f"Generated {num_examples} training examples.")
        return np.array(inputs), np.array(outputs)
    
    def train_model(self, model, criterion, optimizer, inputs, outputs, epochs):
        """
        Trains a machine learning model using the given inputs, outputs, model, criterion, optimizer, and number of epochs.
        
        Args:
            model (torch.nn.Module): The model to be trained.
            criterion (torch.nn.Module): The loss function used for training.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
            inputs (List[torch.Tensor]): The input tensors for training.
            outputs (List[torch.Tensor]): The target output tensors for training.
            epochs (int): The number of epochs to train the model.
        
        Returns:
            List[float]: The average loss for each epoch during training.
        """
        model.train()
        previous_average_loss = None  # Initialize a variable to store the loss of the previous epoch
        loss_difference = "N/A"  # Initialize loss_difference for the first epoch
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(range(len(inputs)), desc=f"Epoch {epoch+1}/{epochs}")
            
            for i in progress_bar:
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32)
                output_tensor = torch.tensor(outputs[i], dtype=torch.float32)
                
                optimizer.zero_grad()
                predictions = model(input_tensor)
                loss = criterion(predictions, output_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            average_loss = total_loss / len(inputs)
            
            if previous_average_loss is not None:
                loss_difference = previous_average_loss - average_loss
                progress_bar.set_postfix({'Avg Loss': f'{average_loss:.4f}', 'Loss Change': f'{loss_difference:.4f}'})
                losses.append(average_loss)
            else:
                progress_bar.set_postfix({'Avg Loss': f'{average_loss:.4f}'})
            
            previous_average_loss = average_loss  # Update the previous_average_loss for the next epoch
            print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}, Loss Change from Previous Epoch: {loss_difference}')
        return losses
            

    
    def simulate_games(self, model, num_games=20, max_guesses=50):
        """
        Simulates a number of games and calculates the average number of guesses per game and overall average similarity.

        Parameters:
            model (torch.nn.Module): The model to use for prediction.
            num_games (int): The number of games to simulate. Default is 20.
            max_guesses (int): The maximum number of guesses allowed per game. Default is 50.

        Returns:
            None
        """
        print("Simulating games...")
        model.eval()
        total_guesses = 0
        total_average_similarity = 0  # To track the average similarity per game
        progress_bar = tqdm(range(num_games), desc=f"Simulating {num_games} games")
        
        for _ in progress_bar:
            target_word = np.random.choice(self.word_list)
            similarities = []  # List to keep track of similarity scores in the current game
            guessed_word = np.random.choice(self.word_list)
            
            for i in range(max_guesses):
                total_guesses += 1
                guessed_vector = self.model[guessed_word].reshape(1, -1)
                similarity = self._get_similarity_score(guessed_word, target_word)
                similarities.append(similarity)  # Add the similarity score to the list
                
                input_vector = np.concatenate((guessed_vector.flatten(), [similarity]))
                input_tensor = torch.tensor(input_vector, dtype=torch.float32)
                with torch.no_grad():
                    predicted_embedding = model(input_tensor).cpu().numpy()
                guessed_word = self.find_closest_word(predicted_embedding)
                
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
        
        

    