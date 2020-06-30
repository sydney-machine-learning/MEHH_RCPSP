# parallelgeneticprogrammingscheduling
Parallel genetic programming for constraint project scheduling


### Instance
Each instance of the benchmark is represented as an object which contains within itself all the variables and information of the problem instance

The class instance when initialised with a filepath of a psplib problem instance reads all information from the file. The parameter combination type, instance type etc are all automatically determined from the filename and the correspomding values are read and stored
Next the precedence relations are read from the file and stored in a typical adjacency list format.
Then the recource consumption and duration of each job is read.

NOTE: All arrays are converted to 1 type indexing i.e arr[0] is dummy value
Therefore arr[job_no] will give the value corresponding to the job_no

Finally the networkx and pygraphviz library are used to draw the graph for visualisation (if required)

### Requirements
matplotlib, networkx, pygraphviz are the only dependencies and all are required for visualisation purposes
They can be installed with `pip3 install <module_name>`

