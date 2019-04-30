import numpy as np
import math
from itertools import permutations

def twopointdistance(a,b):
    x_dif=a[0]-b[0]
    y_dif=a[1]-b[1]
    return math.sqrt((x_dif**2)+(y_dif**2))

def total_distance(a):
    disresult=0
    for i in range(len(a)-1):
        disresult+=twopointdistance(a[i],a[i+1])
    return disresult
# def permute(lst):
#     if len(lst)==0:
#         return []
#     elif len(lst)==1:
#         return[lst]
#     else:
#         l=[]
#         for i in range(len(lst)):
#             x=lst[i]
#             xs=[]
#             for j in range(i):
#                 xs.append(lst[j])
#             for j in range(len(lst)-i-1):
#                 xs.append(lst[i+j+1])
#             for p in permute(xs):
#                 l.append([x]+p)
#         return l
def calculatebest(inputlist):
    outputlist=[]
    result=0
    temp = 0
    # perm=permutations(inputlist)
    for i in permutations(inputlist):
        temp=total_distance(list(i))
        
       
        if result ==0:
            result=temp
        elif  temp<result:
            result = temp
            outputlist=[]
            outputlist.append(list(i))
        elif temp==result:
            outputlist.append(list(i)) 
        
        
    return outputlist,result


n = 3
step= 1
height =300
width =300
with open('data.txt','w') as f:
    for _ in range(step):
        linedata=[]
        traildata=np.random.randn(n,2)
        for i in range(len(traildata)):
            traildata[i][0]=traildata[i][0]*height
            traildata[i][1]=traildata[i][1]*width
        linedata.append(traildata)
        bestline,bestresult=calculatebest(traildata)
        linedata.append(bestline)
        linedata.append(bestresult)
        print(linedata)
        f.write(str(linedata))
for i in permutations([1,2,3,4,5]):
    print(i)