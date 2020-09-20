import numpy as np
log_base_path="./logs/map_elites/set_0/"
file=open(log_base_path+"data_and_charts/new_results.txt","r")
lines=file.readlines()
file.close()
all_aggregate=[]
lin=2
while(lin<len(lines)):
    print(lines[lin])
    all_aggregate.append(float(list(lines[lin].split('%'))[1]))
    lin+=5


print("All aggregates : ",all_aggregate)
all_aggregate=np.array(all_aggregate)
print("Mean ",np.mean(all_aggregate))
print("Median", np.median(all_aggregate))
print("STD",np.std(all_aggregate))
print("MIN",np.min(all_aggregate))
print("MAX",np.max(all_aggregate))
file=open(log_base_path+'final_stats_map_elites_2.txt',"w")
data= "All aggregates : "+str(all_aggregate)+"\nMean  "+str(np.mean(all_aggregate))+"\nMedian  "+str(np.median(all_aggregate))+"\nSTD  "+str(np.std(all_aggregate))+"\nMIN  "+str(np.min(all_aggregate))+"\nMAX  "+str(np.max(all_aggregate))
file.write(data)
file.close()