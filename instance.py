import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir
import pickle
import time
import statistics
from utils import read_param,add_lists,sub_lists, less_than, min_finish_time, find_index,normalised #Utility functions

#Instance definition
class instance(object):
    """
        This is a class for a problem instance and contains all the necessary information and methods for scheduling
    """
 
 
    def __init__(self,filepath="",use_precomputed=True):
        """
        The constructor for instance class
        
        Should be initialised with the filepath to input file
        NOTE: All arrays are converted to 1 type indexing i.e arr[0] is a dummy value, therefore arr[job_no] will give the value corresponding to the job_no
        arr[1] corresponds to the dummy job which is part of the rcpsp and has 0 duration
        """
        
        self.filepath=filepath
        self.n_jobs=0 #Including supersource and sink
        self.horizon=0 #Sum of all durations
        self.k=0 #Number of resources
        self.rel_date=0
        self.due_date=0
        self.tardcost=0
        self.mpm_time=0
        self.adj=[[]] #Adjacency matrix NOTE: adj is 1 indexed adj[0] is dummy node
        self.durations=[0] #List storing durations
        self.job_resources=[[]] #Resource consumed for each job
        self.total_resources=[] #Total availabe resources
        self.nc=0.0 #Network complexity
        self.rf=0.0 #Resource factor
        self.rs=0.0 #Resource strength
        self.parameter_number=0 #Parameter combination number as indicated in param.txt 
        self.instance_number=0#Instance number for a particular parameter combination
        self.instance_type=''#'j30' / 'j60' / 'j90' / 'j120'
        self.filename_comp=''
        
        filename=list(filepath.split('/'))[-1]
        if filename[0]=='j':
                
            self.filename_comp=filename[:-3]
            filename=list(filename.split('_'))
            
            if(filename[0][1] in ['3','6','9']):#j30,j60,j90 type
                self.parameter_number=int(filename[0][3:])
                self.instance_number=int(list(filename[1].split('.'))[0])
                self.instance_type=filename[0][0:3]

            elif(filename[0][1]=='1'):#j120 type
                self.parameter_number=int(filename[0][4:])
                self.instance_number=int(list(filename[1].split('.'))[0])
                self.instance_type=filename[0][0:4] #j120 type
            else:
                self.parameter_number=0
                self.instance_number=0
                self.instance_type=''
                print("Invalid file name")
            self.nc,self.rf,self.rs=params[self.instance_type][self.parameter_number]
            self.read_data()
        elif filename[-3:]=='rcp':
            self.filename_comp=filename[:-4]
            if(filename[0]=='R'):
                self.instance_type='RG300' 
            else:
                self.instance_type=list(filepath.split('/'))[1]+'/'+list(filepath.split('/'))[2]
            self.read_data_RG()
        else:
            print("Invalid file name")
        self.G=nx.DiGraph()#Create a networkx graph object
        self.G_T=nx.DiGraph()#Create a networkx graph object
        for i in range(1,len(self.adj)):
            for j in self.adj[i]:
                self.G.add_edge(i,j)
                self.G_T.add_edge(j,i)

        self.predecessors=[[]]
        self.successors=[[]]
        for i in range(1,self.n_jobs+1):
            self.predecessors.append(list(self.G.predecessors(i)))
            self.successors.append(list(self.G_T.predecessors(i)))
            
        if(use_precomputed):
            data_file=open("./precomputes/"+self.instance_type+"/"+self.filename_comp,"rb")
            (self.earliest_start_times,self.earliest_finish_times,self.latest_start_times,self.latest_finish_times,self.mts,self.mtp,self.rr,self.avg_rreq,self.min_rreq,self.max_rreq,self.mpm_time)=pickle.load(data_file)
            
        else:
                
            #initialize empty arrays for storing values
            self.earliest_start_times=[0]*(self.n_jobs+1)
            self.earliest_finish_times=[0]*(self.n_jobs+1)
            self.latest_start_times=[0]*(self.n_jobs+1)
            self.latest_finish_times=[0]*(self.n_jobs+1)
            self.num_successors=[0]*(self.n_jobs+1)
            self.mts=[0]*(self.n_jobs+1)
            self.mtp=[0]*(self.n_jobs+1)
            
            
            
            #Calculate LFT,LST,EFT,EST
            self.calculate_lt() # Calculates both LFT and LST
            self.calculate_et() # Calculates both EST and EFT
            self.calculate_mts()
            if(self.instance_type=='RG300' or self.instance_type=='RG30'):
                self.mpm_time=max(self.latest_finish_times)
            self.calulate_activity_attributes() #UNCOMMENT PLSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

       


    def read_data(self):
        """Function for reading data and updating attributes from a fixed format .sm file"""
        file=open(self.filepath,"r")
        lines=file.readlines()
        self.n_jobs=int(list(lines[5].strip().split(':'))[1].strip())
        self.horizon=int(list(lines[6].strip().split(':'))[1].strip())
        self.k=int(list(lines[8].strip().split(':'))[1].strip()[0])
        line=list(map(int,list(lines[14].split())))
        self.rel_date,self.due_date,self.tardcost,self.mpm_time=(line[2],line[3],line[4],line[5])
        for i in range(18,18+self.n_jobs):
            line=list(map(int,list(lines[i].split())))
            n_successors=line[2]
            self.adj.append(line[3:3+n_successors])
        
        for i in range(18+self.n_jobs+4,18+self.n_jobs+4+self.n_jobs):
            line=list(map(int,list(lines[i].split())))
            self.durations.append(line[2])
            self.job_resources.append(line[3:3+self.k+1])
        self.total_resources=list(map(int,list(lines[18+self.n_jobs+4+self.n_jobs+3].split())))
        file.close()
    def read_data_RG(self):
        """Function for reading data and updating attributes from a fixed format .rcp file in RG300 format"""
        file=open(self.filepath,"r")
        lines=file.readlines()
        self.n_jobs=int(list(lines[0].strip().split())[0].strip())
        self.horizon=0
        self.k=int(list(lines[0].strip().split())[1].strip())
        self.total_resources=list(map(int,list(lines[1].split())))
        current_line=2
        for i in range(2,2+self.n_jobs):
            line=list(map(int,list(lines[current_line].split())))
            while(not line):
                current_line+=1
                line=list(map(int,list(lines[current_line].split())))
            # print(line)
            self.durations.append(line[0])
            self.horizon+=line[0]
            self.job_resources.append(line[1:1+self.k])
            num_succesors=line[1+self.k]
            self.adj.append(line[1+self.k+1:])
            num_succesors-=len(line[1+self.k+1:])
            while(num_succesors>0):
                current_line+=1
                line=list(map(int,list(lines[current_line].split())))
                self.adj[i-1]+=line
                num_succesors-=len(line)
            current_line+=1

                
        
        file.close()




    def draw(self):
        """
        Function to draw the precedence relations in the form of a DAG for visualisation purposes
        """
        node_colors=['green'] # Start node is green
        node_sizes=[250] #Start node is bigger
        for i in range(len(self.adj)-3):
            node_colors.append('red')
            node_sizes.append(100)
        node_sizes.append(250) #End node is bigger
        node_colors.append('green') #End node is green
        nx.draw(self.G, pos=nx.nx_agraph.graphviz_layout(self.G),node_color=node_colors, edge_color='b',node_size=node_sizes)
        plt.show()
 
 
 
    def calculate_lt(self):
        """ Calculates values of LFT and LST for each job (Does a serial SGS without considering resource constraints)"""
        scheduled=[0]*(self.n_jobs+1)
        graph=self.G_T
        start_vertex=self.n_jobs        
        scheduled[start_vertex]=1
        for g in range(1,self.n_jobs):
            eligible=[]
            
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]==0):
                    pred_lis=list(graph.predecessors(i))
                    #Consider only precedence relations
                    con=True
                    for j in pred_lis:
                        if scheduled[j]==0:
                            con=False
                            break
                    if(con):
                        eligible.append(i)
            #Pick any job from the eligible set (Here always the first)
            choice=eligible[0]
            pred_lis=list(graph.predecessors(choice))
            max_pred_finish_time=0
            for i in pred_lis:
                max_pred_finish_time=max(self.latest_finish_times[i],max_pred_finish_time)
            #Find the precedence feasible start time and schedule it
            self.latest_start_times[choice]=max_pred_finish_time+1
            self.latest_finish_times[choice]=self.latest_start_times[choice]+self.durations[choice]-1
            scheduled[choice]=1
        #Since we are scheduling in reverse we need to invert times i.e subtract it from makespan
        makespan=max(self.latest_finish_times)
        for i in range(1,len(self.latest_finish_times)):
            self.latest_finish_times[i]=makespan-self.latest_start_times[i]
            self.latest_start_times[i]=self.latest_finish_times[i]-self.durations[i]+1
        self.latest_finish_times[1]=0
        self.latest_start_times[self.n_jobs]=self.latest_finish_times[self.n_jobs]
        

    def calculate_et(self):
        """ Calculates values of EFT and EST for each job (Does a serial SGS without considering resource constraints)"""
        scheduled=[0]*(self.n_jobs+1)
        finish_times=[0]*(self.n_jobs+1)
        graph=self.G
        start_vertex=self.n_jobs        
        scheduled[start_vertex]=1
        finish_times[start_vertex]=0
        for g in range(1,self.n_jobs):
            eligible=[]
            scheduled_list=[]
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]):
                    scheduled_list.append(i)
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]==0):
                    pred_lis=list(graph.predecessors(i))
                    #Consider only precedence relations
                    if(set(pred_lis)<=set(scheduled_list)):
                        eligible.append(i)
            #Pick any job from the eligible set (Here always the first)
            choice=eligible[0]
            pred_lis=list(graph.predecessors(choice))
            max_pred_finish_time=0
            for i in pred_lis:
                max_pred_finish_time=max(finish_times[i],max_pred_finish_time)
            #Find the precedence feasible start time and schedule it
            self.earliest_start_times[choice]=max_pred_finish_time+1
            self.earliest_finish_times[choice]=self.earliest_start_times[choice]+self.durations[choice]-1
            finish_times[choice]=self.earliest_start_times[choice]+self.durations[choice]-1
            scheduled[choice]=1


    def calculate_mts(self):
        """Calculates Total succesors and Total predecessors for each job"""
        for i in range(1,self.n_jobs+1):
            self.mts[i]=len(nx.descendants(self.G,i))
        for i in range(1,self.n_jobs+1):
            self.mtp[i]=len(nx.descendants(self.G_T,i))



    def calculate_grpw(self):
        """Calculates Greatest Rank Position Wight(GRPW) for each job"""
        self.grpw=[0]*(self.n_jobs+1)
        for i in range(1,self.n_jobs+1):
            self.grpw[i]=self.durations[i]
            for j in list(self.G_T.predecessors(i)):
                self.grpw[i]+=self.durations[j]



    def calculate_grd(self):
        self.grd=[0]*(self.n_jobs+1)
        """Calculates Greatest Resource Demand(GRD) for each job"""
        for i in range(1,self.n_jobs+1):
            for j in range(self.k):
                self.grd[i]+=self.durations[i]*self.job_resources[i][j]



    def serial_sgs(self,option='forward',priority_rule='LFT',priorities=[],stat='min'):
        """
            Implements the Serial Schedule Generation Scheme

            Parameters:
                option : Forward or reverse scheduling
                priority _rule : Priority rule used. One of ['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD']
            
            Returns:
                Tuple of (Fractional deviation , makespan)
                Fractional deviation = (makespan-self.mpm_time)/self.mpm_time,makespan
        """
        if priority_rule in ['GRPW','GRD']:
            self.calculate_grpw()
            self.calculate_grd()
        #Initialize arrays to store computed values
        start_times=[0]*(self.n_jobs+1) #Start times of schedule
        finish_times=[0]*(self.n_jobs+1) #Finish times of schedule
        earliest_start=[0]*(self.n_jobs+1) #Earliest precedence feasible start times(Different from EST)
        self.resource_consumption=[[0 for col in range(self.k)] for row in range(self.horizon+1)] #2D array of resource consumption of size n x k
        scheduled=[0]*(self.n_jobs+1) #Boolean array to indicate if job is scheduled
        if(option =='forward'): 
            graph=self.G #If forward scheduling use graph as it is
            start_vertex=1
            predecessors=self.predecessors
        else:#option = reverse 
            graph=self.G_T #If reverse scheduling use transpose of grapj
            start_vertex=self.n_jobs
            predecessors=self.successors
        start_times[start_vertex]=0 #Schedule the first dummy job
        finish_times[start_vertex]=0
        scheduled[start_vertex]=1
        for g in range(1,self.n_jobs): #Perform n-1 iterations (Dummy job already scheduled)
            eligible=[] #List of eligible jobs based on precedence only
           
            #For each unscheduled job check if it is eligible
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]==0):
                    con=True
                    for j in predecessors[i]:
                        if scheduled[j]==0:
                            con=False
                            break
                    if(con):
                        eligible.append(i)
            choice=self.choose(eligible,priority_rule=priority_rule,priorities=priorities) #Choose a job according to some priority rule
            max_pred_finish_time=0 #Find the maximum precedence feasible start time for chosen job
            for i in predecessors[choice]:
                max_pred_finish_time=max(finish_times[i],max_pred_finish_time)
            earliest_start[choice]=max_pred_finish_time+1 #Update the found value in array
            scheduled[choice]=1
            feasible_start_time=self.time_resource_available(choice,earliest_start[choice]) #Find the earliest resource feasible time
            start_times[choice]=feasible_start_time
            finish_times[choice]=feasible_start_time+self.durations[choice]-1 #Update finish time
            for i in range(feasible_start_time,finish_times[choice]+1):
                self.resource_consumption[i]=add_lists(self.resource_consumption[i],self.job_resources[choice]) #Update resource consumption
        makespan=max(finish_times) #Makespan is the max value of finish time over all jobs
        if(option!='forward'):
            for i in range(1,self.n_jobs+1):
                finish_times[i]=makespan-start_times[i]
                start_times[i]=finish_times[i]-self.durations[i]+1
        self.serial_finish_times=finish_times
        return (makespan-self.mpm_time)/self.mpm_time,makespan
 
 
 
    def parallel_sgs(self,option='forward',priority_rule='LFT',priorities=[],stat='min'):

        """
            Implements the Parallel Schedule Generation Scheme

            Parameters:
                option : Forward or reverse scheduling
                priority _rule : Priority rule used. One of ['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD','IRSM','ACS','WCS']
            
            Returns:
                Tuple of (Fractional deviation , makespan)
                Fractional deviation = (makespan-self.mpm_time)/self.mpm_time,makespan
        """
        if priority_rule in ['GRPW','GRD']:
            self.calculate_grpw()
            self.calculate_grd()
        if priority_rule in ['IRSM','WCS','ACS']:
            self.irsm=[0]*(self.n_jobs+1)
            self.wcs=[0]*(self.n_jobs+1)
            self.acs=[0]*(self.n_jobs+1)
        #Initialize arrays to store computed values
        start_times=[0]*(self.n_jobs+1) #Start times of schedule
        finish_times=[0]*(self.n_jobs+1) #Finish times of schedule
        scheduled=[0]*(self.n_jobs+1) #Boolean array to indicate if job is scheduled
        if(option =='forward'): 
            graph=self.G #If forward scheduling use graph as it is
            start_vertex=1
            predecessors=self.predecessors
        else:#option = reverse 
            graph=self.G_T #If reverse scheduling use transpose of grapj
            start_vertex=self.n_jobs
            predecessors=self.successors
        start_times[start_vertex]=0 #Schedule the first dummy job
        finish_times[start_vertex]=0
        scheduled[start_vertex]=1
        active_list=[start_vertex]
        completed=[0]*(self.n_jobs+1)
        completed_list=[]
        current_time=0
        current_consumption=[0]*self.k
        resource_consumption=[[0 for col in range(self.k)] for row in range(self.horizon+2)] 
        #Maintain an active and completed disjoint lists and schedule till all jobs are in one of these lists
        while(len(active_list)+len(completed_list)<self.n_jobs):
            current_time=finish_times[active_list[find_index(active_list,finish_times,'min')]]+1 #Update the current time to the minimum of the finish times in the active list + 1
            #Remove completed jobs from active list add them to completed list and update resource consumption
            removals=[]
            for i in active_list:
                if(finish_times[i]<current_time):
                    completed_list.append(i)
                    completed[i]=1
                    current_consumption=sub_lists(current_consumption,self.job_resources[i])
                    removals.append(i)
            for i in removals:
                active_list.remove(i)
            
            #Find the eligible list by considering both precedence and resource feasibility
            precedence_eligible=[]
            eligible=[]
            for i in range(1, self.n_jobs+1):
                if(scheduled[i]==0):
                    pred_list=predecessors[i]
                    con=True
                    for j in pred_list:
                        if(completed[j]==0):
                            con=False
                            break
                    if(con):
                        precedence_eligible.append(i)
            for i in precedence_eligible:
                if(less_than(self.job_resources[i],sub_lists(self.total_resources,current_consumption))):
                    eligible.append(i)
            #Schedule as many jobs as possible from the eligible list
            while(len(eligible)>0):
                if(len(eligible)>1 and priority_rule in ['IRSM','WCS','ACS']):
                    self.calculate_dynamic_priority_rules(eligible,current_time,current_consumption,active_list,finish_times)  #Calculate the values for irsm,wcs,acs
                choice=self.choose(eligible,priority_rule=priority_rule,priorities=priorities) #choose a job based on priority values
                eligible.remove(choice) #Schedule it, set start/finish times, and remove from eligible set
                if(not less_than(add_lists(current_consumption,self.job_resources[choice]),self.total_resources)):
                    continue
                active_list.append(choice)
                scheduled[choice]=1
                start_times[choice]=current_time
                finish_times[choice]=current_time+self.durations[choice]-1
                current_consumption=add_lists(current_consumption,self.job_resources[choice]) #Update resource consumption
            resource_consumption[current_time]=current_consumption
        makespan=max(finish_times) #Makespan is the max value of finish time over all jobs
        #slack calculation starts
        
        self.slack=0
        prev_consumption=[0]*self.k
        for i in range(0,self.horizon+1):
            if resource_consumption[i]==[0]*self.k:
                resource_consumption[i]=prev_consumption
            else:
                prev_consumption=resource_consumption[i]
        for i in range(1,self.n_jobs):
            curr_slack=0
            least_time=1000000
            for j in self.successors[i]:
                least_time=min(least_time,start_times[j])
            for j in range(finish_times[i]+1,least_time):
                if(less_than(self.job_resources[i],sub_lists(self.total_resources,resource_consumption[j]))):
                    curr_slack+=1
                else:
                    break
            self.slack+=curr_slack

        #slack calculation ends
        if(option!='forward'):#If reverse scheduling invert times
            for i in range(1,self.n_jobs+1):
                finish_times[i]=makespan-start_times[i]
                start_times[i]=finish_times[i]-self.durations[i]+1
        self.parallel_finish_times=finish_times
        return (makespan-self.mpm_time)/self.mpm_time,makespan



    def calculate_dynamic_priority_rules(self,eligible,current_time,current_consumption, active_list,finish_times):
        """
            Calculates IRSM, WCS, ACS priority values

            Parameters: 
                eligible: eligible set of jobs based on both precedence and resource constraints
                current_time: Current time when priorities are being calculated
                current_consumption: Amount of resources being consumed currently
                active_list: List of jobs which are scheduled and currently active
                finish_times: Finish times of each job
        """
        for j in eligible:
            sum_e_vals=0 #Sum of E(i,j) over all i
            max_e_val=0 #Max of E(i,j) over all i
            irsm_val=0 # Max of max(0,E(j,i) -LS_i) over all i
            for i in eligible:
                if(i!=j):
                    irsm_val=max(self.earliest_start(j,i,current_time,current_consumption,active_list,finish_times)-self.latest_start_times[i],irsm_val)
                    curr_e_val=self.earliest_start(i,j,current_time,current_consumption,active_list,finish_times)
                    max_e_val=max(curr_e_val,max_e_val)              
                    sum_e_vals+=curr_e_val
            self.irsm[j]=irsm_val
            self.wcs[j]=self.latest_start_times[j]-max_e_val
            self.acs[j]=self.latest_start_times[j]-(1/(len(eligible)-1))*sum_e_vals



    def earliest_start(self,i,j,current_time,current_consumption, active_list, finish_times):
        """
            Find's the earliest time j can be scheduled if i is scheduled at current_time

            Parameters: 
                i,j : Jobs
                current_time: Current time when priorities are being calculated
                current_consumption: Amount of resources being consumed currently
                active_list: List of jobs which are scheduled and currently active
                finish_times: Finish times of each job
            Returns:
                E(i,j)
        """
        starts=[current_time+self.durations[i]]
        if self.isGFP(i,j):
            pass
        elif self.isCSP(i,j,current_consumption):
            starts.append(current_time)
            
        else:
            new_consumption=[elem for elem in current_consumption]

            new_time=current_time
            finished=[0]*(len(active_list))
            while (not self.isCSP(i,j,new_consumption)):
                for act in active_list:
                    if finish_times[act]==new_time and finished[active_list.index(act)]==0:
                        finished[active_list.index(act)]=1
                        new_consumption=sub_lists(new_consumption,self.job_resources[act])
                new_time+=1
            starts.append(new_time)
        return min(starts)



    def isGFP(self,i,j):
        """Checks if (i,j) is a Generally forbidden pair"""
        return not less_than(add_lists(self.job_resources[i],self.job_resources[j]),self.total_resources)
 


    def isCSP(self,i,j,current_consumption):
        "Checks if (i,j) is a currently schedulable pair(simultaneously)"
        new_consumption=add_lists(self.job_resources[i],self.job_resources[j])
        new_consumption=add_lists(new_consumption,current_consumption)
        return less_than(new_consumption,self.total_resources) 
       
    

    def time_resource_available(self,activity,start_time):
        possible_start=start_time #Iterate through all possible start times until one is found
        while(True):
            possible=True
            for i in range(possible_start,possible_start+self.durations[activity]):
                consumed=[0]*self.k
                for j in range(self.k):
                    #Find the resource consumed if scheduled now
                    consumed[j]=self.resource_consumption[i][j]+self.job_resources[activity][j]
                    if(consumed[j]>self.total_resources[j]):
                        #If it exceeds consider next possible time
                        possible=False
                        break
                if(not possible):
                    break

            if(possible):
                return possible_start
            else:
                possible_start+=1


    
    
    def choose(self,eligible,priority_rule='LFT',priorities=[],stat='min'):

        if(priorities==[]):
            if(priority_rule=='LFT'):
                return eligible[find_index(eligible,self.latest_finish_times,'min')]
            elif(priority_rule=='LST'):
                return eligible[find_index(eligible,self.latest_start_times,'min')]
            elif(priority_rule=='EST'):
                return eligible[find_index(eligible,self.earliest_start_times,'min')]
            elif(priority_rule=='EFT'):
                return eligible[find_index(eligible,self.earliest_finish_times,'min')]
            elif(priority_rule=='FIFO'):
                return sorted(eligible)[0]
            elif(priority_rule=='RAND'):
                return random.choice(eligible)
            elif(priority_rule=='SPT'):
                return eligible[find_index(eligible,self.durations,'min')]
            elif(priority_rule=='MTS'):
                return eligible[find_index(eligible,self.mts,'max')]
            elif(priority_rule=='GRPW'):
                return eligible[find_index(eligible,self.grpw,'max')]
            elif(priority_rule=='GRD'):
                return eligible[find_index(eligible,self.grd,'max')]
            elif(priority_rule=='IRSM'):
                return eligible[find_index(eligible,self.irsm,'min')]
            elif(priority_rule=='WCS'):
                return eligible[find_index(eligible,self.wcs,'min')]
            elif(priority_rule=='ACS'):
                return eligible[find_index(eligible,self.acs,'min')]
            else:
                print("Invalid priority rule")
        else:
            if (isinstance(priorities[0], list)):
                votes={}

                for prio in priorities:
                    candidate=eligible[find_index(eligible,prio,stat)]
                    if(candidate not in votes):
                        votes[candidate]=0
                    votes[candidate]+=1
                return max(votes,key=votes.get)
            else:
                return eligible[find_index(eligible,priorities,stat)]
    
    
    
    def calulate_activity_attributes(self):
        """
            Normalises and calculates the activity attributes required for GP
        """
        self.earliest_start_times=normalised(self.earliest_start_times)
        self.earliest_finish_times=normalised(self.earliest_finish_times)
        self.latest_start_times=normalised(self.latest_start_times)
        self.latest_finish_times=normalised(self.latest_finish_times)
        self.mts=normalised(self.mts,self.n_jobs-1)
        self.mtp=normalised(self.mtp,self.n_jobs-1)
        self.rr=[0]*(self.n_jobs+1)
        self.avg_rreq=[0]*(self.n_jobs+1)
        self.min_rreq=[0]*(self.n_jobs+1)
        self.max_rreq=[0]*(self.n_jobs+1)
        for i in range(1,self.n_jobs+1):
            count=0
            sumv=0
            minv=self.job_resources[i][0]/self.total_resources[0]
            maxv=self.job_resources[i][0]/self.total_resources[0]
            for j in range(self.k):
                val=self.job_resources[i][j]
                sumv+=(val/self.total_resources[j])
                minv=min(minv,val/self.total_resources[j])
                maxv=max(maxv,val/self.total_resources[j])
                if(val>0):
                    count+=1
            self.rr[i]=count/self.k
            self.avg_rreq[i]=sumv/self.k
            self.min_rreq[i]=minv
            self.max_rreq[i]=maxv

    
    
    def __str__(self):
        info="Instance Type : " + self.instance_type+"\nParameter number : "+str(self.parameter_number)+"\nInstance number : "+str(self.instance_number)+"\n"
        info+="#jobs : "+str(self.n_jobs)+"\nResources available "+str(self.total_resources)+ " Horizon : "+str(self.horizon)
        return info

params={'j30':[[]],'j60':[[]],'j90':[[]],'j120':[[]]}



read_param('./j30/param.txt',params['j30'],48)
read_param('./j60/param.txt',params['j60'],48)
read_param('./j90/param.txt',params['j90'],48)
read_param('./j120/param.txt',params['j120'],60)



series_priority_rules=['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD']
parallel_priority_rules=['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD','IRSM','ACS','WCS']
types=['j30','j60','j90','j120','RG300']

# series_priority_rules=['LST']
types=['RG30/set1','RG30/set2','RG30/set3','RG30/set4','RG30/set5']

# print(x)
if __name__ == '__main__':
    # statistics.get_stats(instance,series_priority_rules,types,'parallel','forward',use_precomputed=False)
    pass
    

