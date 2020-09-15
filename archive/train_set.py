

train_set=[]

jumps={'set1':(100,900),'set2':(20,180),'set3':(20,240),'set4':(20,240),'set5':(20,240)}

for i in jumps:
    for j in range(1,jumps[i][1],jumps[i][0]):
        train_set.append('./RG30/'+i+'/Patterson'+str(j)+".rcp")
for i in range(1,49):
    train_set.append("./j30/j30"+str(i)+"_1.sm")
    train_set.append("./j60/j60"+str(i)+"_1.sm")
    train_set.append("./j90/j90"+str(i)+"_1.sm")
print(len(train_set))