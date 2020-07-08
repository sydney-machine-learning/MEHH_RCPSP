import time
from os import listdir
from instance import instance
def get_stats(priority_rules,types,mode='serial',option='forward'):
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
                x=instance(i)
                if(mode=='parallel'):
                    y=x.parallel_sgs(option=option,priority_rule=rule)
                elif mode=='serial':
                    y=x.serial_sgs(option=option,priority_rule=rule)
                else:
                    print("Invalid mode")
                total_dev+=y[0]
                total_makespan+=y[1]
                print(i,y,(100*total_dev)/count)

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
