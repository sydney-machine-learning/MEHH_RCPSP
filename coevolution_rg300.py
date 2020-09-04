import random
from deap import base,creator,tools,algorithms,gp
import sys
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
from os import listdir
import pickle
import os
import time
import instance
import statistics
import numpy as np
import operator
import math
import time
import multiprocessing

if not os.path.exists('./logs/coevolution'):
    os.makedirs('./logs/coevolution/training_logs')
if not os.path.exists('./evolved_funcs/coevolution'):
    os.makedirs('./evolved_funcs/coevolution')
#Generate the training set
validation_set=[]
test_set=[]

train_set=["./"+"j30"+'/'+i for i in listdir('./'+"j30") if i!='param.txt']
test_set=[]
for typ in ["RG300"]:
    test_set+=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']

from params_coevolution import *

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
            individuals : The individuals whose fitness is being evaluated by voting
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
toolbox.register("get_best", tools.selBest, k=1)

if __name__ == "__main__":
    all_aggregate=[]
    
    print("Using",os.cpu_count(),"cores")
    for run in range(N_RUNS):
        print("Run #"+str(run))
        pool = multiprocessing.Pool(os.cpu_count())
        toolbox.register("map", pool.map)
        
        # Statistics calculated by evaluating GP
        file=open('./logs/coevolution/species_logs_'+str(run)+".txt","w")
        file.close()
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


        species = [toolbox.species() for _ in range(NUM_SPECIES)]
        



        species_index = list(range(NUM_SPECIES))
        last_index_added = species_index[-1]
        
        # Init with random a representative for each species
        representatives = [random.choice(species[i]) for i in range(NUM_SPECIES)]
        best_fitness_history = [None] * IMPROVMENT_LENGTH
        
        file=open('./logs/coevolution/species_logs_'+str(run)+".txt","a+")
        
       
        g=0 
        while g < NUM_GENERATIONS:
            # Initialize a container for the next generation representatives
            next_repr = [None] * len(species)
            for (i, s), j in zip(enumerate(species), species_index):
                # Vary the species individuals
                s = algorithms.varOr(s, toolbox, LAMBDA,MATING_PROB, MUTATION_PROB)
                # Get the representatives excluding the current species
                r = representatives[:i] + representatives[i+1:]
                for ind in s:
                    # Evaluate and set the individual fitness
                    ind.fitness.values = toolbox.evaluate([ind]+r)
                
                record = mstats.compile(s)
                logbook.record(gen=g, species=j, evals=len(s), **record)
                
                
                print(logbook.stream)
                
                # Select the individuals
                species[i] = toolbox.select(s, len(s))  # Tournament selection
                next_repr[i] = toolbox.get_best(s)[0]   # Best selection
                
             
                
                g += 1
            
            representatives = next_repr
            
            # Keep representatives fitness for stagnation detection
            best_fitness_history.pop(0)
            best_fitness_history.append(representatives[0].fitness.values[0])
            
            try:
                diff = best_fitness_history[-1] - best_fitness_history[0]
            except TypeError:
                diff = float("inf")
            # file.write("Generation :"+str(g)+"\n")
            # file.write("Validation set accuracy of representative "+str(evalCoEvolve(representatives,validation_set))+"\n")
            # file.write("Test set accuracy of representative "+str(evalCoEvolve(representatives,test_set))+"\n")
            
        # val_res=evalCoEvolve(representatives,validation_set)
        test_res=evalCoEvolve(representatives,test_set)
        print(test_res)
        all_aggregate.append(100*test_res[0])
        # print("Results on validation ",val_res)
        print("Results on test ",100*test_res)
        file.write("\n\n\nFinal results : \n")
        # file.write("Validation set accuracy of representative "+str(val_res)+"\n")
        file.write("Test set accuracy of representative "+str(100*test_res)+"\n")
        
        file.close()
        print("\n\nCurrent aggregates : ",all_aggregate)
        current_aggregates=np.array(all_aggregate)
        print("Mean ",np.mean(current_aggregates))
        print("Median", np.median(current_aggregates))
        print("STD",np.std(current_aggregates))
        print("MIN",np.min(current_aggregates))
        print("MAX",np.max(current_aggregates))

        
        # Store the hof in a pickled file
        file=open('./evolved_funcs/coevolution/coevolved_species_'+str(run),'wb')
        pickle.dump(species,file)
        file.close()
        file=open('./logs/coevolution/training_logs/training_log_coevolution_'+str(run)+".txt",'w')
        file.write(str(logbook))
        file.close()


    
    print("All aggregates : ",all_aggregate)
    all_aggregate=np.array(all_aggregate)
    print("Mean ",np.mean(all_aggregate))
    print("Median", np.median(all_aggregate))
    print("STD",np.std(all_aggregate))
    print("MIN",np.min(all_aggregate))
    print("MAX",np.max(all_aggregate))
    file=open('./logs/coevolution/final_stats_coevolution.txt',"w")
    data= "All aggregates : "+str(all_aggregate)+"\nMean  "+str(np.mean(all_aggregate))+"\nMedian  "+str(np.median(all_aggregate))+"\nSTD  "+str(np.std(all_aggregate))+"\nMIN  "+str(np.min(all_aggregate))+"\nMAX  "+str(np.max(all_aggregate))
    file.write(data)
    file.close()
