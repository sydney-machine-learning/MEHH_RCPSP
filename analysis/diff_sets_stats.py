sets=["set_3","set_0","set_1","set_2"]
sizes=[125, 1000, 3375, 8000]
ins=["j60","j90","j120"]
metrics=['B','M','W']
dicts={"set_0":[12,1,4],"set_1":[15,18,3],"set_2":[28,4,8],"set_3":[14,19,28]}
start=[2,7,2,2]

for k in range(len(metrics)):
    
    for i in range(len(sets)):    
        path="../logs/map_elites/"+sets[i]
        print("$MEHH_{"+str(sizes[i])+", "+metrics[k]+"}$",end=" & ")
        for j in ins:
            file=open(path+"/"+j+"_final.txt","r")
            lines=file.readlines()
            file.close()
            line = ""
            if(i==1): 
                for l in range(6):
                    line+=lines[l].strip()
                    line+=" "
                line=list(map(float, list(list(line.split('['))[1].strip()[:-1].strip().split())))
                
            else:
                line =lines[0].strip()
                line=list(map(float,list(list(line.split('['))[1].strip()[:-1].split(','))))
            assert (len(line)==31)
            print("{0:.2f}".format(round(line[dicts[sets[i]][k]],2)),end= ' & ')
            # print(round(line[dicts[sets[i]][k]],2),end = ' & ')
        print("\\\\ ")