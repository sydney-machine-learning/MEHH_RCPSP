from sklearn.ensemble import RandomForestRegressor
import instance
import numpy as np
train_set=[]
validation_set=[]
test_set=[]
types=['j30','j60']
for typ in types:
  for i in range(1,49):
    validation_set.append("./"+typ+"/"+typ+str(i)+"_3.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_1.sm")
    train_set.append("./"+typ+'/'+typ+str(i)+"_2.sm")
    for j in range(4,11):
        test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")
    
for typ in ['j90']:    
    for i in range(1,49):

        for j in range(1,11):
            test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")
for typ in ['j120']:    
    for i in range(1,61):

        for j in range(1,11):
            test_set.append("./"+typ+'/'+typ+str(i)+"_"+str(j)+".sm")
X_train=[]
y_train=[]
for i in range(len(train_set)):
    file=train_set[i]
    inst=instance.instance(file,use_precomputed=True)
    for j in range(1,inst.n_jobs+1):
        tmp=[inst.earliest_start_times[j],inst.earliest_finish_times[j],inst.latest_start_times[j],inst.latest_finish_times[j],inst.mtp[j],inst.mts[j],inst.rr[j],inst.avg_rreq[j],inst.max_rreq[j],inst.min_rreq[j]]
    X_train.append(tmp)
    y_train.append(inst.mpm_time)
X_train=np.array(X_train)
y_train=np.array(y_train)
regr = RandomForestRegressor(max_depth=6, random_state=0)
regr.fit(X_train,y_train)
print(regr.score(X_train,y_train))