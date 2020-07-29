
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pygraphviz as pgv
from qdpy.algorithms.deap import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import gp
import statistics
import operator
import instance
import os
import numpy as np
import random
import warnings
import scipy
import instance
import pickle


#Generate the training set
train_set=[]
validation_set=[]
types=['j30','j60']
for typ in types:
  for i in range(1,49):
    validation_set.append("./"+typ+"/"+typ+str(i)+"_3.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")

# Parameters
n_runs=2
nb_features = 5                            # The number of features to take into account in the container
nb_bins = [7,7,7,7]
features_domain = [(1, 63),(0.7,1.2),(0.1,0.4),(0,450)]      # The domain (min/max values) of the features
fitness_domain = [(0., 1.0)]               # The domain (min/max values) of the fitness
init_batch_size = 1024                     # The number of evaluations of the initial batch ('batch' = population)
batch_size = 1024                           # The number of evaluations in each subsequent batch
nb_iterations = 25                        # The number of iterations (i.e. times where a new batch is evaluated)
cxpb = 0.8
mutation_pb = 0.2                        # The probability of mutating each value of a genome
max_items_per_bin = 1                      # The number of items in each bin of the grid
verbose = True                             
show_warnings = True                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
log_base_path = './logs'

SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HEIGHT_LIMIT = 6 # Height Limit for tree
GEN_MIN_HEIGHT=3
GEN_MAX_HEIGHT=6
"""Eval mode
0:  Evaluate only best individual on train set
1: Evaluate best individual on validation set
2: Evaluate all individuals on grid
"""

eval_mode=[1]


# Update seed



def div(left, right): # Safe division to avoid ZeroDivisionError
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# Generate the primitive set which contains all operators
pset = gp.PrimitiveSet("MAIN",10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
#Rename all arguments
pset.renameArguments(ARG0="ES")
pset.renameArguments(ARG1="EF")
pset.renameArguments(ARG2="LS")
pset.renameArguments(ARG3="LF")
pset.renameArguments(ARG4="TPC")
pset.renameArguments(ARG5="TSC")
pset.renameArguments(ARG6="RR")
pset.renameArguments(ARG7="AvgRReq")
pset.renameArguments(ARG8="MaxRReq")
pset.renameArguments(ARG9="MinRReq")



creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Define the Fitness type(minimisation) 
# weights=(-1,) indicates that there is 1 fitness value which has to be minimised
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # Define Individual type (PrimitiveTree)
def evalSymbReg(individual):

    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    hard_cases_sum=0
    hard_cases_count=0
    serial_cases_sum=0
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=False)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])
        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
        frac2,makespan2=inst.serial_sgs(option='forward',priority_rule='',priorities=priorities)
        if(inst.rs==0.2 and inst.rf==1.0):
            hard_cases_count+=1
            hard_cases_sum+=frac
        serial_cases_sum+=frac2
    fitness=[sumv/len(train_set)]
    features = [len(individual),hard_cases_sum/hard_cases_count,serial_cases_sum/len(train_set),inst.slack]
    return [fitness, features]



# Toolbox defines all gp functions such as mate,mutate,evaluate
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit size of operator tree
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))


# with open("./logs/final0.p", "rb") as f:
#     data = pickle.load(f)


# print(data.keys())
# print(data['container'].best)

# grid=data['container']
ind='add(add(add(sub(sub(AvgRReq,MaxRReq),AvgRReq),LS),add(div(max(TSC,AvgRReq),neg(LF)),LS)),add(LS,mul(min(max(RR,LS),add(RR,MaxRReq)),sub(LF,AvgRReq))))'
test_type=['j30','j60','j90','j120']
sum_total_dev=0
sum_counts=0
print("Individual : ", ind)
# print(len(grid),data.keys())
# print(data["logbook"])
# log_file=open('results_log.txt','a+')
for typ in test_type:
    total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=ind),inst_type=typ,mode='parallel',option='forward',verbose=False)
    print(typ,total_dev_percent,makespan)
    
    # log_file.write(str(grid.best)+" : \n               "+typ+"         "+str(len(train_set))+"               "+str(nb_iterations)+"          "+str(cxpb)+"           "+str(mutation_pb)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n")
    
    sum_total_dev+=total_dev
    sum_counts+=count
# log_file.write("Aggregate % : "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
# log_file.close()
print("Aggregate % ",(sum_total_dev*100)/sum_counts)
