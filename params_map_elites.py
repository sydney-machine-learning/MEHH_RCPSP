 
n_runs=31
nb_features = 3                            # The number of features to take into account in the container
nb_bins = [16,16,16]
features_domain = [(4, 63),(0.75,1.2),(0.15,0.4)]      # The domain (min/max values) of the features
fitness_domain = [(0., 1.0)]               # The domain (min/max values) of the fitness
init_batch_size = 1024                     # The number of evaluations of the initial batch ('batch' = population)
batch_size = 1024                           # The number of evaluations in each subsequent batch
nb_iterations = 25                        # The number of iterations (i.e. times where a new batch is evaluated)
cxpb = 0.8
mutation_pb = 0.2                        # The probability of mutating each value of a genome
test_threshold=26                           # %deviation below which an individual is printed
max_items_per_bin = 1                      # The number of items in each bin of the grid
verbose = True                             
show_warnings = True                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
log_base_path = './logs/map_elites/data_and_charts'

SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HEIGHT_LIMIT = 6 # Height Limit for tree
GEN_MIN_HEIGHT=2
GEN_MAX_HEIGHT=5
"""Eval mode
0:  Evaluate only best individual on train set
1: Evaluate best individual on validation set
2: Evaluate all individuals on grid
"""

eval_mode=[1,2]