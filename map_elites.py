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

from deap import base,creator,tools,algorithms,gp
import statistics
import operator
import instance
import os
import numpy as np
import random
import warnings
import scipy
import instance
import time
from utils import sub_lists
from multiprocessing import Pool
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
from params_map_elites import *


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
    features = [len(individual),hard_cases_sum/hard_cases_count,serial_cases_sum/len(train_set)]
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
def parallelised_evaluation(ind):
    global log_file
    
    sum_total_dev=0
    sum_counts=0
    for typ in test_type:
        total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=ind),inst_type=typ,mode='parallel',option='forward',verbose=False)
        sum_total_dev+=total_dev
        sum_counts+=count
    log_file.write("Individual : "+str(ind)+"\n")
    log_file.write("Aggregate % : "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
    if((sum_total_dev*100)/sum_counts < 26.5):
        print("\n\nIndividual : ",ind)
        print("Aggregate % ",(sum_total_dev*100)/sum_counts)
        print("Fitness ",ind.fitness)
        print("Features ",ind.features) 

if __name__ == "__main__":
    all_aggregate=[]
    for run in range(n_runs):
        print("Run #"+str(run))
        seed = 1001+run
        np.random.seed(seed)
        random.seed(seed)
        # Create a dict storing all relevant infos
        results_infos = {}
        results_infos['features_domain'] = features_domain
        results_infos['fitness_domain'] = fitness_domain
        results_infos['nb_bins'] = nb_bins
        results_infos['init_batch_size'] = init_batch_size
        results_infos['nb_iterations'] = nb_iterations
        results_infos['batch_size'] = batch_size

        grid = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list)
        
        #(none, multiprocessing, concurrent, multithreading, scoop)
        with ParallelismManager("concurrent", toolbox=toolbox) as pMgr:
            # Create a QD algorithm
            algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_size = init_batch_size, batch_size = batch_size, niter = nb_iterations,
                    cxpb = cxpb, mutpb = mutation_pb,
                    verbose = verbose, show_warnings = show_warnings, results_infos = results_infos, log_base_path = log_base_path)
            algo.final_filename="final"+str(run)+".p"
            # Run the illumination process !
            algo.run()

        # Print results info
        print(f"Total elapsed: {algo.total_elapsed}\n")
        print(grid.summary())
        file=open('./logs/grid_logs/grid_summary_'+str(run),"w")
        file.write(grid.summary())
        file.close()


        print("Evaluating best individual")
        print("Function ", grid.best)
        print("best fitness:", grid.best.fitness)
        print("best features:", grid.best.features)
        file=open('./evolved_funcs/map_elites_grid_'+str(run),"wb")
        pickle.dump(grid,file)
        file.close()
        file=open('./evolved_funcs/map_elites_best_'+str(run),"wb")
        pickle.dump(grid.best,file)
        file.close()
        if 0 in eval_mode:
            test_type=['j30','j60','j90','j120']
            sum_total_dev=0
            sum_counts=0
            print("Best Individual on Training: ", grid.best)
            log_file=open('map_elites_results_log.txt','a+')
            for typ in test_type:
                total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=grid.best),inst_type=typ,mode='parallel',option='forward',verbose=False)
                print(typ,total_dev_percent,makespan)
                
                log_file.write(str(grid.best)+" : \n               "+typ+"         "+str(seed)+"               "+str(nb_iterations)+"          "+str(cxpb)+"           "+str(mutation_pb)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n\n")
                
                sum_total_dev+=total_dev
                sum_counts+=count
            log_file.write("Aggregate % (best on train): "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
            log_file.close()
            print("Aggregate % ",(sum_total_dev*100)/sum_counts)
        if 1 in eval_mode:
            print("\n\nEvaluating all individuals on validation set....")
            min_deviation=999
            
            for ind in grid:
                
                total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=ind),mode='parallel',option='forward',use_precomputed=True,verbose=False)
                # print("Deviation of individual ",total_dev_percent,min_deviation)
                if total_dev_percent<min_deviation:
                    min_deviation=total_dev_percent
                    best_individual=ind
            
            test_type=['j30','j60','j90','j120']
            sum_total_dev=0
            sum_counts=0
            log_file=open('map_elites_results_log.txt','a+')
            print("Best Individual on validation: ", best_individual)
            for typ in test_type:
                total_dev_percent,makespan,total_dev,count=statistics.evaluate_custom_rule(instance.instance,toolbox.compile(expr=best_individual),inst_type=typ,mode='parallel',option='forward',verbose=False)
                print(typ,total_dev_percent,makespan)
                
                log_file.write(str(best_individual)+" : \n               "+typ+"         "+str(seed)+"               "+str(nb_iterations)+"          "+str(cxpb)+"           "+str(mutation_pb)+"         "+str(round(total_dev_percent,2))+"        "+str(makespan)+"       \n\n")
                
                sum_total_dev+=total_dev
                sum_counts+=count
            log_file.write("Aggregate % (best on validation): "+str((sum_total_dev*100)/sum_counts)+"  \n\n\n")
            log_file.close()
            all_aggregate.append((sum_total_dev*100)/sum_counts)
            print("Aggregate % ",(sum_total_dev*100)/sum_counts)
            print("Fitness ",best_individual.fitness)
            print("Features ",best_individual.features)
            total_dev_percent,total_makespan,total_dev,count=statistics.evaluate_custom_set(validation_set,instance.instance,toolbox.compile(expr=best_individual),mode='parallel',option='forward',use_precomputed=True,verbose=False)
            print("Performance on validation ",total_dev_percent)
        
        if 2 in eval_mode:
            print("\n\nEvaluating all individuals on grid.....")
            test_type=['j30','j60','j90','j120']
            log_file=open('map_elites_results_log.txt','a+')
            with Pool(os.cpu_count()) as p:
                p.map(parallelised_evaluation,grid)
               

            log_file.close()
        # Generate and Store graph
        nodes, edges, labels = gp.graph(grid.best)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw("./gp_trees/individual"+"__map_elites"+str(run) + ".png")

        # Create plots
        plot_path = os.path.join(log_base_path, "performancesGrid"+str(run)+".pdf")
        plotGridSubplots(grid.quality_array[... ,0], plot_path, plt.get_cmap("nipy_spectral"), grid.features_domain, grid.fitness_extrema[0], nbTicks=None)
        print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))

        plot_path = os.path.join(log_base_path, "activityGrid"+str(run)+".pdf")
        plotGridSubplots(grid.activity_per_bin, plot_path, plt.get_cmap("nipy_spectral"), grid.features_domain, [0, np.max(grid.activity_per_bin)], nbTicks=None)
        print("\nA plot of the activity grid was saved in '%s'." % os.path.abspath(plot_path))

        print("All results are available in the '%s' pickle file." % algo.final_filename)
    print("All aggregates : ",all_aggregate)
    all_aggregate=np.array(all_aggregate)
    print("Mean ",np.mean(all_aggregate))
    print("Median", np.median(all_aggregate))
    print("STD",np.std(all_aggregate))
    print("MIN",np.min(all_aggregate))
    print("MAX",np.max(all_aggregate))
    file=open('./logs/final_stats_map_elites.txt',"w")
    data= "All aggregates : "+str(all_aggregate)+"\nMean  "+str(np.mean(all_aggregate))+"\nMedian  "+str(np.median(all_aggregate))+"\nSTD  "+str(np.std(all_aggregate))+"\nMIN  "+str(np.min(all_aggregate))+"\nMAX  "+str(np.max(all_aggregate))
    file.write(data)
    file.close()


"""
none : 51sec
multiprocessing : 17
concurrent : 18.84
multithreading : NA
scoop : 
"""
"""
4 features no slack: 64
5 features no slack calc:68
5 features slack calc: 
"""