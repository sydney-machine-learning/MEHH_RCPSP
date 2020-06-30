import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv

j30_params=[[]]
j60_params=[[]]
j90_params=[[]]
j120_params=[[]]
def read_param(filepath,dest,n_lines=48):

    file=open(filepath,"r")
    lines=file.readlines()
    for i in range(n_lines):
        line=list(map(float,list(lines[i].strip().split())))
        dest.append(line[1:])
    file.close()
read_param('./j30/param.txt',j30_params,48)
read_param('./j60/param.txt',j60_params,48)
read_param('./j90/param.txt',j90_params,48)
read_param('./j120/param.txt',j120_params,60)

class instance(object):
    def __init__(self,filepath=""):
        #NOTE: All arrays are converted to 1 type indexing i.e arr[0] is dummy value
        #Therefore arr[job_no] will give the value corresponding to the job_no
        self.filepath=filepath
        self.n_jobs=0 #Including supersource and sink
        self.horizon=0
        self.k=0 #Number of resources
        self.rel_date=0
        self.due_date=0
        self.tardcost=0
        self.mpm_time=0
        self.adj=[[]] #Adjacency matrix NOTE: adj is 1 indexed adj[0] is dummy node
        self.durations=[0]
        self.job_resources=[[]] #Resource consumed for each job
        self.total_resources=[] #Total availabe resources
        self.nc=0.0 #Network complexity
        self.rf=0.0 #Resource factor
        self.rs=0.0 #Resource strength
        self.parameter_number=0 #Parameter combination number as indicated in param.txt 
        self.instance_number=0#Instance number for a particular parameter combination
        self.instance_type=''
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

    def read_data(self):
        file=open(self.filepath,"r")
        lines=file.readlines()
        
        self.n_jobs=int(self.split_clean_colon(lines[5])[1])
        self.horizon=int(self.split_clean_colon(lines[6])[1])
        self.k=int(list(self.split_clean_colon(lines[8])[1].split())[0])
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
    def split_clean_colon(self,line):
        line=list(line.strip().split(':'))
        line[0]=line[0].strip()
        line[1]=line[1].strip()
        return line
    def draw(self):
        G=nx.DiGraph()
        for i in range(1,len(self.adj)):
            for j in self.adj[i]:
                G.add_edge(i,j)
        node_colors=['green']
        node_sizes=[250]
        for i in range(len(self.adj)-3):
            node_colors.append('red')
            node_sizes.append(100)
        node_sizes.append(250)
        node_colors.append('green')
        nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G),node_color=node_colors, edge_color='b',node_size=node_sizes)
        plt.show()
    def __str__(self):
        info="Instance Type : " + self.instance_type+"\nParameter number : "+str(self.parameter_number)+"\nInstance number : "+str(self.instance_number)+"\n"
        info+="#jobs : "+str(self.n_jobs)+"\nResources available "+str(self.total_resources)
        return info

filepath="./j30/j3048_8.sm"

x=instance(filepath)

print(x)
x.draw()