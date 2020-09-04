from os import listdir
import instance
import pickle
import sys
types=['j30','j60','j90','j120','RG300']
for typ in types:
    all_files=["./"+typ+'/'+i for i in listdir('./'+typ) if i!='param.txt']
    for file in all_files:
        x=instance.instance(file,use_precomputed=False)
        
        filename=list(file.split('/'))[-1][:-3]
        if(filename[0]=='R'):
            filename=filename[:-1]
        data=(x.earliest_start_times,x.earliest_finish_times,x.latest_start_times,x.latest_finish_times,x.mts,x.mtp,x.rr,x.avg_rreq,x.min_rreq,x.max_rreq,x.mpm_time)
        dumpfile=open("./precomputes/"+typ+"/"+filename,"wb")
        pickle.dump(data,dumpfile)
        print(file,"                  ", end='\r')
        sys.stdout.flush()