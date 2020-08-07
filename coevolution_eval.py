import random
from deap import base,creator,tools,algorithms,gp
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from os import listdir
import pickle
import time
import instance
import statistics
import numpy as np
import operator
import math
import time
import multiprocessing

#Generate the training set
train_set=[]
validation_set=[]
test_set=[]
types=['j30','j60']
for typ in types:
  for i in range(1,49):
    validation_set.append("./"+typ+"/"+typ+str(i)+"_3.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")
    for j in range(4,11):
        test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")
    
for typ in ['j90']:    
    for i in range(1,49):

        for j in range(1,11):
            test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")
for typ in ['j120']:    
    for i in range(1,61):

        for j in range(1,11):
            test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")


SPECIES_SIZE = 1024
NUM_SPECIES = 5

IMPROVMENT_TRESHOLD = 0.5
IMPROVMENT_LENGTH = 5
EXTINCTION_TRESHOLD = 5.0
# Parameters for GP
N_RUNS=1

NUM_GENERATIONS=25 # Number of generation to evolve
MATING_PROB=0.8 # Probability of mating two individuals
MUTATION_PROB=0.2 # Probability of introducing mutation
SELECTION_POOL_SIZE=7 # Number of individuals for tournament
HOF_SIZE=3 # Number of top individuals to evaluate on test set
HEIGHT_LIMIT = 6 # Height Limit for tree
GEN_MIN_HEIGHT=2
GEN_MAX_HEIGHT=5
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

def evalCoEvolve(individuals,train_set):
    """
        Evaluation function which calculates fitness of an individual
        Parameters: 
            individual : The individual whose fitness is being evaluated
        Returns:
            A tuple of fitness values since only 1 value is used we return a 1 element tuple (fitness,)
    """
    funcs=[]
    for individual in individuals:
        func = toolbox.compile(expr=individual) # Transform the tree expression in a callable function
        funcs.append(func)
    sumv=0 # Evaluate the individual on each file and train set and return the normalised sum of deviation values
    for i in range(len(train_set)):
        file=train_set[i]
        inst=instance.instance(file,use_precomputed=True)
        priorities=[]
        for func in funcs:
            tmp_priorities=[0]*(inst.n_jobs+1)
            
            for j in range(1,inst.n_jobs+1):
                tmp_priorities[j]=func(inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j])
            priorities.append(tmp_priorities)
        frac,makespan=inst.parallel_sgs(option='forward',priority_rule='',priorities=priorities)
        sumv+=frac
    return (sumv/len(train_set),)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("species", tools.initRepeat, list, toolbox.individual, SPECIES_SIZE)


toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalCoEvolve,train_set=train_set)
toolbox.register("select", tools.selTournament, tournsize=SELECTION_POOL_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=GEN_MIN_HEIGHT, max_=GEN_MAX_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorators to limit size of operator tree
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=HEIGHT_LIMIT))
toolbox.register("get_best", tools.selBest, k=3)

if __name__ == "__main__":
    all_aggregate=[]
    for run in range(N_RUNS):
        print("Run #"+str(run))
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        
        # Statistics calculated by evaluating GP
        
        mstats = tools.Statistics(lambda ind: ind.fitness.values)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = "gen", "species", "evals", "std", "min", "avg", "max"
        # Update seed
        seed = 1000+run
        np.random.seed(seed)
        random.seed(seed)

        file=open('./evolved_funcs/gp/coevlov_species3_'+str(run),'rb')
        species=pickle.load(file)
        file.close()
        all_individuals=[]
        for spec in species:
            tmp=toolbox.get_best(spec)[0]
            all_individuals+=[tmp]
        print(evalCoEvolve(spec,test_set))

        # # Evaluate on test set and generate tree structure
       
        # for spec in species:
        #     min_deviation=999
        #     total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=ind),mode='parallel',option='forward',use_precomputed=True,verbose=False)
        #     if total_dev_percent<min_deviation:
        #         min_deviation=total_dev_percent
        #         best_individual=ind
        # print("Best Individual Run_"+str(run)+" :  ", best_individual)
        # test_type=['j30','j60','j90','j120']
        # sum_total_dev=0
        # sum_counts=0
        # log_file=open('./logs/gp/gp_results_log.txt','a+')
        # for typ in test_type:
        #     total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=best_individual),inst_type=typ,mode='parallel',option='forward',verbose=False)
        #     print(typ,total_dev_percent,makespan)
        #     log_file.write(str(best_individual)+" : \n               "+typ+"         "+str(seed)+"               "+str(NUM_GENERATIONS)+"          "+str(MATING_PROB)+"           "+str(MUTATION_PROB)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n\n")
            
        #     sum_total_dev+=total_dev
        #     sum_counts+=count
        # log_file.write("Aggregate % : "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
        # log_file.close()
        # print("Aggregate % ",(sum_total_dev*100)/sum_counts)

        # all_aggregate.append((sum_total_dev*100)/sum_counts)

        # # # Generate and Store graph
        # # nodes, edges, labels = gp.graph(best_individual)
        # # g = pgv.AGraph()
        # # g.add_nodes_from(nodes)
        # # g.add_edges_from(edges)
        # # g.layout(prog="dot")
        # # for i in nodes:
        # #     n = g.get_node(i)
        # #     n.attr["label"] = labels[i]
        # # g.draw("./gp_trees/"+str(round((sum_total_dev*100)/sum_counts,2))+"_run_"+str(run)+".png")
    
    # print("All aggregates : ",all_aggregate)
    # all_aggregate=np.array(all_aggregate)
    # print("Mean ",np.mean(all_aggregate))
    # print("Median", np.median(all_aggregate))
    # print("STD",np.std(all_aggregate))
    # print("MIN",np.min(all_aggregate))
    # print("MAX",np.max(all_aggregate))
    # file=open('./logs/gp/final_stats_gp.txt',"w")
    # data= "All aggregates : "+str(all_aggregate)+"\nMean  "+str(np.mean(all_aggregate))+"\nMedian  "+str(np.median(all_aggregate))+"\nSTD  "+str(np.std(all_aggregate))+"\nMIN  "+str(np.min(all_aggregate))+"\nMAX  "+str(np.max(all_aggregate))
    # file.write(data)
    # file.close()
# (0.2581039472958451,)
#(0.25650638217126775,)