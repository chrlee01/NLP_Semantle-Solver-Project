# NLP_Semantle-Solver-Project
Our attempts at creating a solver for the game Semantle using 3 different methods
## Project Members:
- Christian Lee
- Eric Wilmes
- Coleman Kisicki
- Reid King
- Jay Nasser

## Solvers:
- Non-ML solver: A guessing algorithm that doesn't use any kind of machine learning to solve the game. This model is the best one we made.
- Deep-Q-Learning Solver: A solver that trains itself by playing lots of games and learns from the results of the actions it choses.
- Neural Network Solver: A simple neural network that is trained on generated example data to produce good guesses.
- Random Solver: A base solver that simply guesses random words. We only used this as a baseline to compare our results to.

## Instructions:
- install requirements by running ```pip install -r requirements.txt``` with a console open to the project folder
- For the NN solver and the Deep learning solver, there is a driver file that can be run. This will train and evaluate the performance of the model.
- For the non-ML solver the file eric_solver.py can be run.
- To get the base results to compare the solvers to the RandomSolver.py can be run