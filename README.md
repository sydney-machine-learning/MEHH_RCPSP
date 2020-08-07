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

![demo](demo.png?raw=true "Demo image")


Member functions:

init(): The constructor for instance class. Should be initialised with filepath to .sm file

read_data() : Reads data from file and updates all attributes

draw(): visualise precedence relations

calculate_et,lt,mts,grpw,grd(): Calculates the priority values for each rule

Serial_sgs(): implements the serial schedule generation scheme

Parallel_sgs(): implements the parallel schedule generation scheme

calculate _dynamic_priority_rules(): calculates the priority functions such as irsm,wcs,acs which can not be precomputed and computed at schedule time

Calculate_activity_attributes(): Normalises and calculates activity attributes which will be used as attributes in the GP 

Earliest_start(): Calculates E(i,j). The first time j can be scheduled if i is scheduled at the current time

## Results
### Priority rules
### Serial SGS
% Deviation
     j30    j60    j90    j120    
EST  24.63  23.82  23.24  60.55  
EFT  27.59  26.59  25.79  64.38  
LST  19.95  17.45  16.07  46.74  
LFT  20.81  18.13  16.67  48.11  
SPT  34.21  33.99  32.16  77.94  
FIFO  25.19  23.31  21.81  57.79  
MTS  21.84  18.99  17.59  50.32  
RAND  29.65  28.74  27.38  68.38  
GRPW  26.26  26.02  24.99  65.28  
GRD  28.13  27.89  26.96  68.42

Makespan
       j30     j60     j90     j120     
EST  31069  42940  51300  91164  
EFT  31798  43882  52371  93341  
LST  29905  40699  48299  83274  
LFT  30077  40931  48533  84039  
SPT  33379  46403  54962  100942  
FIFO  31179  42711  50656  89496  
MTS  30342  41208  48912  85239  
RAND  32280  44542  52953  95521  
GRPW  31438  43626  51971 93693  
GRD  31865  44243  52765  95515

### Parallel SGS
% Deviation

     j30    j60    j90    j120    
     
EST  23.11  21.68  21.45  55.78  
EFT  23.66  22.46  21.82  55.77  
LST  19.31  17.12  15.80  44.04  
LFT  19.05  17.46  15.90  43.86  
SPT  25.59  23.77  23.60  60.33  
FIFO  22.10  20.38  19.47  51.57  
MTS  19.69  17.98  16.70  46.03  
RAND  25.42  23.02  23.30  60.01  
GRPW  23.28  22.32  21.83  57.87  
GRD  24.96  24.11  23.18  61.30  
IRSM  19.15  17.90  16.41  46.36   
ACS  18.77  16.92  15.78  43.71    
WCS  18.44  16.88  15.37  43.53    

Makespan

       j30     j60     j90     j120 
       
EST  30696  42199  50554  88471  
EFT  30844  42480  50720  88472  
LST  29757  40596  48191  81753  
LFT  29688  40724  48238  81653  
SPT  31287  42911  51431  91012  
FIFO  30442  41735  49689  86008  
MTS  29865  40906  48563  82863  
RAND  31239  42648  51264  90817  
GRPW  30724  42402  50660  89546  
GRD  31131  42982  51223  91464  
IRSM  29708  40880  48446  83087  
ACS  29626  40540  48188  81569  
WCS  29551  40523  48015  81465  
Time taken :  1901.825050354004



### GP
Using GP to evolve the priority rules we get a lower % deviation than any of the priority rules

j30 18.33

j60 17.12

j90 15.37

j120 42.89

Evolved function tree 

![tree](gp_trees/42.88__2.png?raw=true "GP tree")
