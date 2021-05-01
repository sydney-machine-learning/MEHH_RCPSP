import time
from os import listdir
import sys
import pickle
def get_stats(instance,priority_rules,types,mode='serial',option='forward',use_precomputed=True,custom_set={},verbose=True):
    """
        Prints the percentage deviation and makespan for all priority rules and instance types specified
        Parameters:
            instance : instance class
            priority_rules : All priority rules for which stats to be calculated
            types : All types for which stats to be calculated
    """
    start=time.time()
    ans={'j30':{},'j60':{},'j90':{},'j120':{},'RG300':{},'RG30/set1':{},'RG30/set2':{},'RG30/set3':{},'RG30/set4':{},'RG30/set5':{}}
    for typ in types:
        
        if typ in custom_set:
            all_files=custom_set[typ]
        else :
            all_files=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']
        
        for rule in priority_rules:
            total_dev=0;
            total_makespan=0
            count=0
            for i in all_files:
                
                try:
                    x=instance(i,use_precomputed=use_precomputed)
                    if(mode=='parallel'):
                        y=x.parallel_sgs(option=option,priority_rule=rule)
                    elif mode=='serial':
                        y=x.serial_sgs(option=option,priority_rule=rule)
                    else:
                        print("Invalid mode")
                except Exception as e:
                    print("Encountered error while reading",i)
                    print(e)
                    continue
                count+=1
                total_dev+=y[0]
                total_makespan+=y[1]
                if verbose:
                    print(i,y,(100*total_dev)/count,"                        ",end='\r')
                    sys.stdout.flush()
            total_dev_percent=(100*total_dev)/len(all_files)
            if(verbose):
                print()
                print(typ,rule,total_dev_percent,total_makespan)
            ans[typ][rule]=[total_dev_percent,total_makespan]
    if not verbose:
        return total_dev_percent,total_makespan
    print(ans)
    print("% Deviation")
    print('     ',end='')
    for i in types:
        print(i,end='    ')
        if(i!=types[-1]):
            print(' & ',end='')
    print()
    for i in priority_rules:
        print(i,end=' & ')
        for j in types:
            print("%.2f"%ans[j][i][0],end='  ')
            if(j!=types[-1]):
                print(' & ',end='')
        print(" \\\\")
    print("Makespan")
    print('       ',end='')
    for i in types:
        print(i,end='     ')
        if(i!=types[-1]):
            print(' & ',end='')
    print()
    for i in priority_rules:
        print(i,end=' & ')
        for j in types:
            print(ans[j][i][1],end='  ')
            if(j!=types[-1]):
                    print(' & ',end='')
            print(" \\\\")
    print()
    for i in priority_rules:
        print(i,end=' & ')
        for j in types:
            print("%.2f"%ans[j][i][0],end=' & ')
            print(ans[j][i][1],end='')
            if(j!=types[-1]):
                print(' & ',end='')
        print(" \\\\")
    file=open('results','wb')
    pickle.dump(ans,file)
    file.close()
    print("Time taken : ",time.time()-start) 

def evaluate_custom_rule(instance,priority_func,inst_type='j120',mode='parallel',option='forward',use_precomputed=True,verbose=True):
    """
        Evaluates custom priority rule which is given by priority_func
        Parameters:
            instance : instance class
            priority_func : Function which evaluates priority value from activity attributes
            types : All types for which stats to be calculated
    """
    all_files=["./"+inst_type+'/'+i for i in listdir('./'+inst_type) if i!='param.txt']
    all_files.sort()
    total_dev=0;
    total_makespan=0
    count=0
    for file in all_files:
        filename=list(file.split('/'))[-1]
        # print(filename)
        if (file[-4] in ['1','2','3']) and filename[1] in ['3','6']: #Ignore train set and validation set i.e all instances with j30xx_1,j30xx_2,j30xx_3,j60xx_1,j60xx_2,j60xx_3
            continue
        count+=1
        x=instance(file,use_precomputed=use_precomputed)
        priorities=[]
        priorities=[0]*(x.n_jobs+1)
        for job in range(1,x.n_jobs+1):
            priorities[job]=priority_func(x.earliest_start_times[job],x.earliest_finish_times[job],x.latest_start_times[job],x.latest_finish_times[job],x.mtp[job],x.mts[job],x.rr[job],x.avg_rreq[job],x.max_rreq[job],x.min_rreq[job])
        if(mode=='parallel'):
            y=x.parallel_sgs(option=option,priority_rule='',priorities=priorities)
        elif mode=='serial':
            y=x.serial_sgs(option=option,priority_rule='',priorities=priorities)
        else:
            print("Invalid mode")
        total_dev+=y[0]
        total_makespan+=y[1]
        if(verbose):
            print(file,y,(100*total_dev)/count,"                   ", end='\r')
            sys.stdout.flush()
    if(verbose):
        print()
        print(count, inst_type, "files read")
    total_dev_percent=(100*total_dev)/count
    return (total_dev_percent,total_makespan,total_dev,count)


def evaluate_custom_set(eval_set,instance,priority_func,mode='parallel',option='forward',use_precomputed=True,verbose=True):
    """
    """
    eval_set.sort()
    total_dev=0
    total_makespan=0
    count=0
    for file in eval_set:
        filename=list(file.split('/'))[-1]
        count+=1
        x=instance(file,use_precomputed=use_precomputed)
        priorities=[]
        priorities=[0]*(x.n_jobs+1)
        for job in range(1,x.n_jobs+1):
            priorities[job]=priority_func(x.earliest_start_times[job],x.earliest_finish_times[job],x.latest_start_times[job],x.latest_finish_times[job],x.mtp[job],x.mts[job],x.rr[job],x.avg_rreq[job],x.max_rreq[job],x.min_rreq[job])
        if(mode=='parallel'):
            y=x.parallel_sgs(option=option,priority_rule='',priorities=priorities)
        elif mode=='serial':
            y=x.serial_sgs(option=option,priority_rule='',priorities=priorities)
        else:
            print("Invalid mode")
        total_dev+=y[0]
        total_makespan+=y[1]
        if(verbose):
            print(file,y,(100*total_dev)/count,"                   ", end='\r')
            sys.stdout.flush()
    if(verbose):
        print()
        print(len(eval_set)," files read")
    
    total_dev_percent=(100*total_dev)/count
    return (total_dev_percent,total_makespan,total_dev,count)


