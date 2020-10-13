""" Convert Map elite log to a pickle file with all data"""
import pickle
import numpy as np
file=open("./../logs/map_elites/set_3/map_elites_results_log.txt","r")
lines=file.readlines()
file.close()
data={}

run=0
for i in range(len(lines)):
    line=lines[i].strip()
    if "Validation" in line:
        data[run]=dict()
        ind=lines[i+2].strip()
        lis=list(lines[i+3].strip().split())
        print(run,lis[-2],lis[-3])
        # print(ind)
        data[run]['ind']=ind
        
        data[run]['dev']=float(lis[-3 + (run==30)])
        data[run]['makespan']=int(lis[-2 + (run==30)])
        
        
        run+=1
print(data)
file=open("./../map_elites_data_3","wb")
pickle.dump(data,file)
file.close()

file=open("./../map_elites_data_3","rb")
data_new=pickle.load(file)
print(data_new)
makespans=[]
devs=[]
for i in data_new:
    makespans.append(data[i]['makespan'])
    devs.append(data[i]['dev'])

print("All aggregates : ",makespans)
makespans=np.array(makespans)
print("Mean ",np.mean(makespans))
print("Median", np.median(makespans))
print("STD",np.std(makespans))
print("MIN",np.min(makespans))
print("MAX",np.max(makespans))

print("All aggregates : ",devs)
devs=np.array(devs)
print("Mean ",np.mean(devs))
print("Median", np.median(devs))
print("STD",np.std(devs))
print("MIN",np.min(devs))
print("MAX",np.max(devs))