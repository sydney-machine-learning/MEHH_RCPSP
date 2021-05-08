
import os

# Parameters for GP
N_RUNS=31
POP_SIZE=1024
NUM_GENERATIONS=25 # Number of generation to evolve
MATING_PROB=0.8 # Probability of mating two individuals
MUTATION_PROB=0.2 # Probability of introducing mutation
SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HOF_SIZE=1 # Number of top individuals to evaluate on test set
HEIGHT_LIMIT = 7 # Height Limit for tree
MU=1024 # The number of individuals to select for the next generation.
LAMBDA=1024 # The number of children to produce at each generation.
GEN_MIN_HEIGHT=2
GEN_MAX_HEIGHT=5

occupied=0 # Set number for storing results
filename="./gp.py"
with open(filename) as infile:
    exec(infile.read())