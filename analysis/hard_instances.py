import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir
import pickle
import time
import instance
import statistics
import numpy as np
from deap import base,creator,tools,algorithms,gp
import operator
import math
import time 
import multiprocessing

train_set=['./datasets/j30/'+i for i in listdir('./datasets/j30') if i!="param.txt"]
validation_set=[]
for i in range(1,480,10):
    validation_set.append("./datasets/RG300/datasets/RG300_"+str(i)+".rcp")
all_rg300=["./datasets/RG300/"+i for i in listdir('./datasets/RG300')]
test_set=[i for i in all_rg300 if i not in validation_set]

hard_starts=[101,141,261,301,421,461]
hard_test_tmp=[]
for i in hard_starts:
    for j in range(i,i+20):
        hard_test_tmp.append("./datasets/RG300/datasets/RG300_"+str(j)+".rcp")

hard_test=[i for i in hard_test_tmp if i not in validation_set]


def div(left, right): # Safe division to avoid ZeroDivisionError
    try:
        return left / right
    except ZeroDivisionError:
        return 1


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
    """
        Evaluation function which calculates fitness of an individual
        Parameters: 
            individual : The individual whose fitness is being evaluated
        Returns:
            A tuple of fitness values since only 1 value is used we return a 1 element tuple (fitness,)
    """
    start=time.time()
    func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=True)
        priorities=[0]*(inst.n_jobs+1)
        for j in range(1,inst.n_jobs+1):
            priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])

        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
    return (sumv/len(train_set),)

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












setlabels=["gp_data","map_elites_data_3","map_elites_data_0","map_elites_data_1","map_elites_data_2"]
all_all_aggregates=[]
all_all_aggregates_makespan=[]

# for setlabel in setlabels:
#     file=open("./"+setlabel,'rb')
#     data=pickle.load(file)
#     file.close()
#     all_aggregate=[]
#     all_aggregate_makespan=[]
#     for run in range(N_RUNS):
#         min_deviation=100000


    
        
#         best_individual = data[run]['ind']


#         total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(hard_test,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
#         print("Aggregate % ",total_dev_percent)
#         print("Makespan ",total_makespan )
#         all_aggregate.append(total_dev_percent)
#         all_aggregate_makespan.append(total_makespan)
        

   

#     print("All aggregates : ",all_aggregate)
#     all_aggregate=np.array(all_aggregate)
#     print("Mean ",np.mean(all_aggregate))
#     print("Median", np.median(all_aggregate))
#     print("STD",np.std(all_aggregate))
#     print("MIN",np.min(all_aggregate))
#     print("MAX",np.max(all_aggregate))

#     all_all_aggregates.append(all_aggregate)
#     all_all_aggregates_makespan.append(all_aggregate_makespan)
file = open("hard_instances_data","rb")
# data  = {"deviation":all_all_aggregates,"makespan":all_all_aggregates_makespan}
# pickle.dump(data,file)
data = pickle.load(file)
all_all_aggregates = data["deviation"]
all_all_aggregates_makespan = data["makespan"]
file.close()
names=["Mean","Median", "Best", "Worst", "St. Dev."]
metrics=[np.mean,np.median,np.min,np.max,np.std]
for i in range(len(names)):
    func=metrics[i]
    print(names[i],end='')
    for lis in all_all_aggregates:
        print(' & ',end='')
        print(round(func(lis),2),end='')

    print("\\\\")
print("\n"*3)
metrics2 = [np.min,np.median,np.max]
names2=["B","M","W"]
caps = ["$GPHH$-","$MEHH_{125}$-","$MEHH_{1000}$-","$MEHH_{3375}$-","$MEHH_{8000}$-"]

for j in range(len(names2)):
    for i in caps:
        lis = all_all_aggregates[caps.index(i)]
        lis2 = all_all_aggregates_makespan[caps.index(i)]
        print(i+names2[j]+" & "+str(round(metrics2[j](lis),2))+" & "+str(int(metrics2[j](lis2)))+" \\\\ ")


"""
$GPHH$-B & 2219.69 & 161659 \\ 
$GPHH$-M & 2224.90 & 161998 \\ 
$GPHH$-W & 2234.72 & 162631 \\ 
$MEHH_{125}$-B & 2218.42 & 161551 \\ 
$MEHH_{125}$-M & 2219.55 & 161684 \\ 
$MEHH_{125}$-W & 2230.40 & 162246 \\ 
$MEHH_{1000}$-B & 2218.64 & 161576 \\ 
$MEHH_{1000}$-M & 2220.31 & 161656 \\ 
$MEHH_{1000}$-W & 2227.37 & 162196 \\ 
$MEHH_{3375}$-B & 2217.96 & 161502 \\ 
$MEHH_{3375}$-M & 2220.16 & 161656 \\ 
$MEHH_{3375}$-W & 2227.69 & 162220 \\ 
$MEHH_{8000}$-B & 2217.70 & 161523 \\ 
$MEHH_{8000}$-M & 2220.33 & 161722 \\ 
$MEHH_{8000}$-W & 2224.58 & 161935 \\ 
"""