import time
from os import listdir
# from instance import instance
import sys
import pickle
def get_stats(instance,priority_rules,types,mode='serial',option='forward',use_precomputed=True):
    start=time.time()
    ans={'j30':{},'j60':{},'j90':{},'j120':{}}
    for typ in types:
        all_files=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']
        for rule in priority_rules:
            total_dev=0;
            total_makespan=0
            count=0
            for i in all_files:
                count+=1
                x=instance(i,use_precomputed=use_precomputed)
                if(mode=='parallel'):
                    y=x.parallel_sgs(option=option,priority_rule=rule)
                elif mode=='serial':
                    y=x.serial_sgs(option=option,priority_rule=rule)
                else:
                    print("Invalid mode")
                total_dev+=y[0]
                total_makespan+=y[1]
                print(i,y,(100*total_dev)/count,"                        ",end='\r')
                sys.stdout.flush()
            print()
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

def evaluate_custom_rule(instance,priority_func,inst_type='j120',mode='serial',option='forward',use_precomputed=True):
    
    all_files=["./"+inst_type+'/'+i for i in listdir('./'+inst_type) if i!='param.txt']
    
    total_dev=0;
    total_makespan=0
    count=0
    for file in all_files:
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
        
        print(file,y,(100*total_dev)/count,"                  ", end='\r')
        sys.stdout.flush()
    print()
    total_dev_percent=(100*total_dev)/len(all_files)
    return (total_dev_percent,total_makespan)

