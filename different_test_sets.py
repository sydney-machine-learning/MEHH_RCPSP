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

train_set=[]
sets={'j60':{'val':[],'test':[]},'j90':{'val':[],'test':[]},'j120':{'val':[],'test':[]}}
for i in range(1,49):
    sets['j60']['val'].append("./j60/j60"+str(i)+"_1.sm")
    sets['j90']['val'].append("./j90/j90"+str(i)+"_1.sm")
    for j in range(2,11):
        sets['j60']['test'].append("./j60/j60"+str(i)+"_"+str(j)+".sm")
        sets['j90']['test'].append("./j90/j90"+str(i)+"_"+str(j)+".sm")
for i in range(1,61): 
    sets['j120']['val'].append("./j120/j120"+str(i)+"_1.sm")
    for j in range(2,11):
        sets['j120']['test'].append("./j120/j120"+str(i)+"_"+str(j)+".sm")

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

path="./logs/gp/set_0/data_and_charts/"

for typ in sets:
    validation_set=sets[typ]['val']
    test_set=sets[typ]['test']
    all_aggregate=[]
    all_aggregate_makespan=[]
    for run in range(N_RUNS):
        min_deviation=100000



        file=open(path+'evolved_pop_'+str(run),'rb')
        pop=pickle.load(file)
        file.close()
        best_individual=pop[0]
        perfs={}
        fin=0
        for ind in pop:
            if str(ind) not in perfs:

                total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=ind),mode='parallel',option='forward',use_precomputed=True,verbose=False)
                
            
                if total_dev_percent<min_deviation:
                    min_deviation=total_dev_percent
                    best_individual=ind
                perfs[str(ind)]=total_dev_percent
            fin+=1
            

    

        total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(test_set,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
        print("Aggregate % ",total_dev_percent)
        print("Makespan ",total_makespan )
        all_aggregate.append(total_dev_percent)
        all_aggregate_makespan.append(total_makespan)
        file=open(path+"../"+typ+"_results.txt","a")
        file.write("Run#"+str(run)+"  "+str(len(perfs))+" unique individuals in population"+"\n")
        file.write(str(best_individual)+"\n")
        file.write("Aggregate% "+str(total_dev_percent)+"\n")
        file.write("Makespan% "+str(total_makespan)+"\n\n")

        file.close()
    print("All aggregates : ",all_aggregate_makespan)
    all_aggregate_makespan=np.array(all_aggregate_makespan)
    print("Mean ",np.mean(all_aggregate_makespan))
    print("Median", np.median(all_aggregate_makespan))
    print("STD",np.std(all_aggregate_makespan))
    print("MIN",np.min(all_aggregate_makespan))
    print("MAX",np.max(all_aggregate_makespan))

    print("All aggregates : ",all_aggregate)
    all_aggregate=np.array(all_aggregate)
    print("Mean ",np.mean(all_aggregate))
    print("Median", np.median(all_aggregate))
    print("STD",np.std(all_aggregate))
    print("MIN",np.min(all_aggregate))
    print("MAX",np.max(all_aggregate))

    file=open(path+"../"+typ+'_final.txt',"a")
    data= "All aggregates : "+str(all_aggregate)+"\nMean  "+str(np.mean(all_aggregate))+"\nMedian  "+str(np.median(all_aggregate))+"\nSTD  "+str(np.std(all_aggregate))+"\nMIN  "+str(np.min(all_aggregate))+"\nMAX  "+str(np.max(all_aggregate))
    data2= "All aggregates makespans: "+str(all_aggregate_makespan)+"\nMean  "+str(np.mean(all_aggregate_makespan))+"\nMedian  "+str(np.median(all_aggregate_makespan))+"\nSTD  "+str(np.std(all_aggregate_makespan))+"\nMIN  "+str(np.min(all_aggregate_makespan))+"\nMAX  "+str(np.max(all_aggregate_makespan))
    file.write(data)
    file.write(data2)
    file.close()