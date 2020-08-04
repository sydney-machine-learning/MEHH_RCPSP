from sklearn.ensemble import RandomForestRegressor
import instance
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
    inst=instance.instance(file,use_precomputed=False)
    