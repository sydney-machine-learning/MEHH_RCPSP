# Map-Elites based Hyper Heuristic for RCPSP
Multidimensional Archive of Phenotypic Elites (MAP-Elites) is a quality diversity based algorithm which constructs an archive of solution based on genotypic and phenotypic features of an individual
In this project the automated evolution of priority rules for generating schedules for the Resource Constrained Project Scheduling Problem(RCPSP) is demonstrated

# Datasets
The repo uses 5 datasets `j30`, `j60`, `j90`, `j120`, `RG300` each having instances with the corresponding number of activities suggested by the name

# Implementation
The [instance](./instance.py) file implements code for basic parsing, scheduling algorithm and priority rules. Running it would generate a table of deviation and makespan values for all the different human designed rules such as LFT, LST, FIFO, etc.

The [gp_rg300](./gp_rg300.py) file implements the classic genetic programming based approach for evolving rules on the `RG300` dataset. The algorithm will automatically parallelise the computations across all the cores available on the CPU.

The [map_elites_rg300](./map_elites_rg300.py) file contains the implementation of MAP-Elites for RCPSP again parallelised across multiple cores.

An example evolved tree ![tree](gp_trees/25.9_run_0.png?raw=true "GP tree")

The `analysis` folder contains all the scripts used to analse, plot, generate results
The `logs` folder contains all the results after running both GP and MAP-Elites for 31 runs and 25 generations each
The `precomputes` folder acts as cache for speeding up certain calculations


### Diversity plot
This plot shows the loss of diversity faced in GP and how MAP-Elites maintains and increases the diversity over multiple generations

![diversity](imgs/coverage_plot_mp_elites.png?raw=true "Diverity plot")

For queries on implementation/dataset contact : 

Kousik Rajesh 

kousik18@iitg.ac.in
