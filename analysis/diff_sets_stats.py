sets=["set_3","set_0","set_1","set_2"]
sizes=[125, 1000, 3375, 8000]
ins=["j60","j90","j120"]
start=[2,7,2,2]
for i in range(len(sets)):
    path="../logs/map_elites/"+sets[i]
    print(str(sizes[i]),end=' & ')
    for j in ins:
        file=open(path+"/"+j+"_final.txt","r")
        lines=file.readlines()
        # print(lines[1:10])
        # print(lines[2:6])
        try:
            print(round(float(list(lines[start[i]+2].split())[1]),2),end=" & ")
            print(round(float(list(lines[start[i]].split())[1]),2),end=" & ")
            print(round(float(list(lines[start[i]+3].split())[1]),2),end=" & ")
        except:
            print("ERROR ",lines[5])
            exit()
    print("\\\\ ")