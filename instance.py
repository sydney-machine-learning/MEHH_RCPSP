import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import random
from os import listdir
import pickle
import time
def read_param(filepath,dest,n_lines=48):

    file=open(filepath,"r")
    lines=file.readlines()
    for i in range(n_lines):
        line=list(map(float,list(lines[i].strip().split())))
        dest.append(line[1:])
    file.close()

def find_index(index_list,value_list,stat='min'):
    if(stat=='min'):
        pos=0
        minv=value_list[index_list[0]]
        for i in range(len(index_list)):
            if(value_list[index_list[i]]<minv or (value_list[index_list[i]]==minv and index_list[i]<index_list[pos])):
                minv=value_list[index_list[i]]
                pos=i
    else: # stat='max'
        pos=0
        maxv=value_list[index_list[0]]
        for i in range(len(index_list)):
            if(value_list[index_list[i]]>maxv or (value_list[index_list[i]]==maxv and index_list[i]<index_list[pos])):
                maxv=value_list[index_list[i]]
                pos=i            

    return pos
class instance(object):
    def __init__(self,filepath=""):
        #NOTE: All arrays are converted to 1 type indexing i.e arr[0] is dummy value
        #Therefore arr[job_no] will give the value corresponding to the job_no
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
        self.instance_type=''#'j30'/'j60'/'j90'/'j120'
        if(filepath):
            filename=list(filepath.split('/'))[-1]
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
                print("Invalid file name")
            
            self.read_data()
            self.G=nx.DiGraph()#Create a networkx graph object
            for i in range(1,len(self.adj)):
                for j in self.adj[i]:
                    self.G.add_edge(i,j)
            self.G_T=nx.reverse(self.G)#Reverse edges to get transpose of graph for reverse scheduling
        #initialize empty arrays for storing values
        self.earliest_start_times=[0]*(self.n_jobs+1)
        self.earliest_finish_times=[0]*(self.n_jobs+1)
        self.latest_start_times=[0]*(self.n_jobs+1)
        self.latest_finish_times=[0]*(self.n_jobs+1)
        self.num_successors=[0]*(self.n_jobs+1)
        self.mts=[0]*(self.n_jobs+1)
        self.grpw=[0]*(self.n_jobs+1)
        self.grd=[0]*(self.n_jobs+1)
        #Calculate LFT,LST,EFT,EST
        self.calculate_lt()
        self.calculate_et()                 
        self.calculate_mts()
        self.calculate_grpw()
        self.calculate_grd()
    def read_data(self):
        #Hardcoded function to read data in the given format
        file=open(self.filepath,"r")
        lines=file.readlines()
        self.n_jobs=int(list(lines[5].strip().split(':'))[1].strip())
        self.horizon=int(list(lines[6].strip().split(':'))[1].strip())
        self.k=int(list(lines[8].strip().split(':'))[1].strip()[0])
        line=list(map(int,list(lines[14].split())))
        self.rel_date,self.due_fate,self.tardcost,self.mpm_time=(line[2],line[3],line[4],line[5])
        for i in range(18,18+self.n_jobs):
            line=list(map(int,list(lines[i].split())))
            n_succesors=line[2]
            self.adj.append(line[3:3+n_succesors])
        
        for i in range(18+self.n_jobs+4,18+self.n_jobs+4+self.n_jobs):
            line=list(map(int,list(lines[i].split())))
            self.durations.append(line[2])
            self.job_resources.append(line[3:3+self.k+1])
        self.total_resources=list(map(int,list(lines[18+self.n_jobs+4+self.n_jobs+3].split())))
        file.close()

    def draw(self):
        #Draws graph of instance
        node_colors=['green']
        node_sizes=[250]
        for i in range(len(self.adj)-3):
            node_colors.append('red')
            node_sizes.append(100)
        node_sizes.append(250)
        node_colors.append('green')
        nx.draw(self.G, pos=nx.nx_agraph.graphviz_layout(self.G),node_color=node_colors, edge_color='b',node_size=node_sizes)
        plt.show()
    def calculate_lt(self):
        #Calculates LFT and LST and updates corresponding class variable 
        # Schedule in the same way as serial SGS without considering the resource constraints
        scheduled=[0]*(self.n_jobs+1)
        graph=self.G_T
        start_vertex=self.n_jobs        
        scheduled[start_vertex]=1
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
                max_pred_finish_time=max(self.latest_finish_times[i],max_pred_finish_time)
            #Find the precedence feasible start time and schedule it
            self.latest_start_times[choice]=max_pred_finish_time+1
            self.latest_finish_times[choice]=self.latest_start_times[choice]+self.durations[choice]-1
            scheduled[choice]=1
        #Since we are scheduling in reverse we need to invert times i.e subtract it from makespan
        makespan=max(self.latest_finish_times)
        for i in range(1,len(self.latest_finish_times)):
            self.latest_finish_times[i]=makespan-self.latest_start_times[i]
            self.latest_start_times[i]=self.latest_finish_times[i]-self.durations[i]
    def calculate_et(self):
        #Calculates EFT and EST and updates corresponding class variable
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
        for i in range(1,self.n_jobs+1):
            self.mts[i]=len(nx.descendants(self.G,i))
    def calculate_grpw(self):
        
        for i in range(1,self.n_jobs+1):
            self.grpw[i]=self.durations[i]
            for j in list(self.G_T.predecessors(i)):
                self.grpw[i]+=self.durations[j]
    def calculate_grd(self):
        for i in range(1,self.n_jobs+1):
            for j in range(self.k):
                self.grd[i]+=self.durations[i]*self.job_resources[i][j]

    def serial_sgs(self,option='forward',priority_rule='LFT'):

        #Initialize arrays to store computed values
        start_times=[0]*(self.n_jobs+1) #Start times of schedule
        finish_times=[0]*(self.n_jobs+1) #Finish times of schedule
        earliest_start=[0]*(self.n_jobs+1) #Earliest precedence feasible start times(Different from EST)
        resource_consumption=[[0 for col in range(self.k)] for row in range(self.horizon+1)] #2D array of resource consumption of size n x k
        scheduled=[0]*(self.n_jobs+1) #Boolean array to indicate if job is scheduled
        
        
        if(option =='forward'): 
            #If forward scheduling use graph as it is
            graph=self.G
            start_vertex=1
        else:#option = reverse 
            #If reverse scheduling use transpose of grapj
            graph=self.G_T
            start_vertex=self.n_jobs
        #Schedule the first dummy job
        start_times[start_vertex]=0
        finish_times[start_vertex]=0
        scheduled[start_vertex]=1
        #Perform n-1 iterations (Dummy job already scheduled)
        for g in range(1,self.n_jobs):
            eligible=[] #List of eligible jobs based on precedence only
            scheduled_list=[] #List of jobs already scheduled
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]):
                    scheduled_list.append(i) #Form the scheduled list using the boolean array
            #For each unscheduled job check if it is eligible
            for i in range(1,self.n_jobs+1):
                if(scheduled[i]==0):
                    pred_lis=list(graph.predecessors(i))
                    if(set(pred_lis)<=set(scheduled_list)): #Job is eligible if its predecossors are a subset of scheduled jobs
                        eligible.append(i)
            choice=self.choose(eligible,priority_rule=priority_rule) #Choose a job according to some priority rule
            pred_lis=list(graph.predecessors(choice)) #find predecessors of chosen job
            max_pred_finish_time=0 #Find the maximum precedence feasible start time for chosen job
            for i in pred_lis:
                max_pred_finish_time=max(finish_times[i],max_pred_finish_time)

            earliest_start[choice]=max_pred_finish_time+1 #Update the found value in array
            feasible_start_time=self.time_resource_available(choice,resource_consumption,earliest_start[choice]) #Find the earliest resource feasible time
            finish_times[choice]=feasible_start_time+self.durations[choice]-1 #Update finish time
            for i in range(feasible_start_time,finish_times[choice]+1):
                for j in range(self.k):
                    resource_consumption[i][j]+=self.job_resources[choice][j] #Update resource consumption
            scheduled[choice]=1
        makespan=max(finish_times) #Makespan is the max value of finish time over all jobs

        if(option!='forward'):
            for i in range(1,len(finish_times)):
                finish_times[i]=makespan-finish_times[i]
        print(finish_times)
        return (makespan-self.mpm_time)/self.mpm_time,makespan
    def parallel_sgs(self,option='forward',priority_rule='LFT'):
        #Initialize arrays to store computed values
        start_times=[0]*(self.n_jobs+1) #Start times of schedule
        finish_times=[0]*(self.n_jobs+1) #Finish times of schedule
        
        scheduled=[0]*(self.n_jobs+1) #Boolean array to indicate if job is scheduled
        
        if(option =='forward'): 
            #If forward scheduling use graph as it is
            graph=self.G
            start_vertex=1
        else:#option = reverse 
            #If reverse scheduling use transpose of grapj
            graph=self.G_T
            start_vertex=self.n_jobs
        #Schedule the first dummy job
        start_times[start_vertex]=0
        finish_times[start_vertex]=0
        scheduled[start_vertex]=1
        
        active_list=[start_vertex]
        completed_list=[]
        current_time=0
        current_consumption=[0]*self.k
        
        while(len(active_list)+len(completed_list)<self.n_jobs):
            
            current_time=finish_times[active_list[find_index(active_list,finish_times,'min')]]+1
            
            # print(current_time)
            for i in active_list:
                if(finish_times[i]<current_time):
                    completed_list.append(i)
                    for j in range(self.k):
                        current_consumption[j]-=self.job_resources[i][j]
            for i in completed_list:
                if i in active_list:
                    active_list.remove(i)
            
            #update resources
            precedence_eligible=[]
            eligible=[]
            for i in range(1, self.n_jobs+1):
                if(scheduled[i]==0):
                    pred_list=list(graph.predecessors(i))
                    if(set(pred_list)<=set(completed_list)):
                        precedence_eligible.append(i)
            for i in precedence_eligible:
                if(self.current_feasible(i,current_consumption)):
                    eligible.append(i)
            while(len(eligible)>0):
                choice=self.choose(eligible,priority_rule=priority_rule)
                eligible.remove(choice)
                if(not self.current_feasible(choice,current_consumption)):
                    continue
                
                active_list.append(choice)
                scheduled[choice]=1
                start_times[choice]=current_time
                finish_times[choice]=current_time+self.durations[choice]-1
                #update resources,eligible?
            
                for j in range(self.k):
                    current_consumption[j]+=self.job_resources[choice][j] #Update resource consumption
                # print(current_consumption)
                
        makespan=max(finish_times) #Makespan is the max value of finish time over all jobs

        if(option!='forward'):
            for i in range(1,len(finish_times)):
                finish_times[i]=makespan-finish_times[i]
        # print(finish_times)
        return (makespan-self.mpm_time)/self.mpm_time,makespan
    def time_resource_available(self,activity,resource_consumption,start_time):
        possible_start=start_time #Iterate through all possible start times until one is found
        while(True):
            possible=True
            for i in range(possible_start,possible_start+self.durations[activity]):
                consumed=[0]*self.k
                for j in range(self.k):
                    #Find the resource consumed if scheduled now
                    consumed[j]=resource_consumption[i][j]+self.job_resources[activity][j]
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

    def time_feasible(self,activity,resource_consumption,time):
        possible=True
        for i in range(time,time+self.durations[activity]):
            consumed=[0]*self.k
            for j in range(self.k):
                #Find the resource consumed if scheduled now
                consumed[j]=resource_consumption[i][j]+self.job_resources[activity][j]
                if(consumed[j]>self.total_resources[j]):
                    #If it exceeds consider next possible time
                    possible=False
                    break
            if(not possible):
                break
        return possible
    def current_feasible(self,activity,current):
        possible=True
        consumed=[0]*self.k
        for i in range(self.k):
            consumed[i]=current[i]+self.job_resources[activity][i]
            if(consumed[i]>self.total_resources[i]):
                #If it exceeds consider next possible time
                possible=False
                break
        return possible
            
    def choose(self,eligible,priority_rule='LFT'):
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
        else:
            print("Invalid priority rule")

    def __str__(self):
        info="Instance Type : " + self.instance_type+"\nParameter number : "+str(self.parameter_number)+"\nInstance number : "+str(self.instance_number)+"\n"
        info+="#jobs : "+str(self.n_jobs)+"\nResources available "+str(self.total_resources)+ " Horizon : "+str(self.horizon)
        return info

j30_params=[[]]
j60_params=[[]]
j90_params=[[]]
j120_params=[[]]


read_param('./j30/param.txt',j30_params,48)
read_param('./j60/param.txt',j60_params,48)
read_param('./j90/param.txt',j90_params,48)
read_param('./j120/param.txt',j120_params,60)


#IRSM,ACS,WCS
priority_rules=['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND']
types=['j30','j60','j90','j120']

x=instance('./j30/j3048_5.sm')

print(x.parallel_sgs(priority_rule='GRPW'))
print(x.grpw)


if __name__ == '__main__':
    start=time.time()
    ans={'j30':{},'j60':{},'j90':{},'j120':{}}
    for typ in types:
        all_files=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']
        for rule in priority_rules:
            total_dev=0;
            total_makespan=0
            
            for i in all_files:
                
                x=instance(i)
                y=x.parallel_sgs(option='forward',priority_rule=rule)
                print(i,y)
                total_dev+=y[0]
                total_makespan+=y[1]
            total_dev_percent=(100*total_dev)/len(all_files)
            print(typ,rule,total_dev_percent,total_makespan)
            ans[typ][rule]=[total_dev_percent,total_makespan]
    print(ans)
    print("% Deviation")
    print('     ',end='')
    for i in types:
        print(i,end='    ')
    print()
    for i in priority_rules:
        print(i,end='  ')
        for j in types:
            print("%.2f"%ans[j][i][0],end='  ')
        print()
    print("Makespan")
    print('       ',end='')
    for i in types:
        print(i,end='     ')
    print()
    for i in priority_rules:
        print(i,end='  ')
        for j in types:
            print(ans[j][i][1],end='  ')
        print()

    file=open('results','wb')
    pickle.dump(ans,file)
    file.close()
    print("Time taken : ",time.time()-start)