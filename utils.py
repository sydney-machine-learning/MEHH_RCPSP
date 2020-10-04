def read_param(filepath,dest,n_lines=48):

    file=open(filepath,"r")
    lines=file.readlines()
    for i in range(n_lines):
        line=list(map(float,list(lines[i].strip().split())))
        dest.append(line[1:])
    file.close()
def add_lists(l1,l2):
    return [sum(x) for x in zip(l1, l2)]

def sub_lists(l1,l2):
    return [a - b for a, b in zip(l1, l2)]
    
def less_than(l1,l2):
    for i in range(len(l1)):
        if(l1[i]>l2[i]):
            return False
    return True
def min_finish_time(ref_time,finish_times):
    mft=10**8+10
    for i in range(len(finish_times)):
        if(finish_times[i]>ref_time):
            mft=min(mft,finish_times[i])
    if(mft==10**8+10):
        mft=ref_time
    return mft
def find_index(index_list,value_list,stat='min'):
    #Function to find the index from index list whose value in value list is minimum or maximum
    if(stat=='min'):
        pos=0
        minv=value_list[index_list[0]]
        for i in range(len(index_list)):
            if(value_list[index_list[i]]<minv or (value_list[index_list[i]]==minv and index_list[i]<index_list[pos])):
                minv=value_list[index_list[i]]
                pos=i
    else: # stat='max'
        pos=0
        maxv=value_list[index_list[0]]
        for i in range(len(index_list)):
            if(value_list[index_list[i]]>maxv or (value_list[index_list[i]]==maxv and index_list[i]<index_list[pos])):
                maxv=value_list[index_list[i]]
                pos=i            

    return pos
def normalised(arr,norm=0):
    if(norm==0):
        newarr=[i/max(arr) for i in arr]
    else:
        newarr=[i/norm for i in arr]
    return newarr
def latex(*argv):
    args=[]
    for arg in argv:
        args.append(arg)
    for arg in args:

        print(arg,end='')
        if(arg!=args[-1]):
            print(" & ",end='')
    print("\\\\")
