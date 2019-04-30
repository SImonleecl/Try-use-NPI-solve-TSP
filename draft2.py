import numpy as np
import math
from itertools import permutations

def I_clear():
    global I
    I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def reset_ptr():#0
    global auc,I,P,data,Rt
    I_clear()
    I[0]=1
    P[0]=1
    Rt=0
    record()
    auc=[0,0,0,0,0]
    P[0]=0
    I_clear()
    I[8]=1
    Rt=1
    record()
def ptr_to_distance():#1
    global auc,data,I,P,Rt
    P[1]=1
    I_clear()
    I[1]=1
    Rt=0
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[1]=0
    I_clear()
    I[6]=1
    record()
    Rt=1
def reset_ptr_4():#2
    global auc,data,I,P,Rt
    P[2]=1
    I_clear()
    I[2]=1
    Rt=0
    record()
    auc[4]=0 
    P[2]=0
    I_clear()
    I[6]=1
    Rt=1
    record()
def twopointdistance():#3
    global auc,data,I,P,Rt
    P[3]=1
    I_clear()
    I[3]=1
    Rt=0
    record()
    if auc[2]+1>=data[0][5]:
        auc[4]+=1
    else:
        x_dif=data[auc[0]][auc[2]][0]-data[auc[0]][auc[2]+1][0]
        y_dif=data[auc[0]][auc[2]][1]-data[auc[0]][auc[2]+1][1]
        data[auc[0]][auc[1]]+=math.sqrt((x_dif**2)+(y_dif**2))
    P[3]=0
    I_clear()
    I[6]=1
    Rt=1
    record()
def shift_2():#4
    global auc,data,I,P,Rt
    P[4]=1
    I_clear()
    I[4]=1
    Rt=0
    record()
    auc[2]+=1
    P[4]=0
    I_clear()
    I[6]=1
    Rt=1
    record()
def reset_ptr_1to4():#5
    global auc,data,I,P,Rt
    P[5]=1
    I_clear()
    I[5]=1
    Rt=0
    record()
    auc[2]=0
    auc[3]=0
    auc[4]=0
    auc[1]=0
    P[5]=0
    I_clear()
    I[6]=1
    Rt=1
    record()
def linedistance():#6
    global auc,data,I,P,Rt
    P[6]=1
    I_clear()
    I[6]=1
    Rt=0
    record()
    while auc[4]==0:
        ptr_to_distance()#i1#i6
    reset_ptr_4()#i2i6
    while auc[4]==0:
        twopointdistance()#i3#i6
        shift_2()#i4#i6
    reset_ptr_1to4()#i5i6
    P[6]=0
    I_clear()
    I[8]=1
    Rt=1
    record()
def shift_0():#7
    global auc,data,I,P,Rt
    P[7]=1
    I_clear()
    I[7]=1
    Rt=0
    record()
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[7]=0
    I_clear()
    I[8]=1
    Rt=1
    record()
def total_distance():#8
    global auc,data,I,P,Rt #i8
    P[8]=1
    I_clear()
    I[8]=1
    Rt=0
    record()
    reset_ptr() #i0 #i8
    if auc[0]<data[0][6]:
        while auc[4]==0:
            linedistance()#i6 #i8
            shift_0()#i7 #i8
    P[8]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def shift_to_distance():#9
    global auc,data,I,P,Rt
    P[9]=1
    I_clear()
    I[9]=1
    Rt=0
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[9]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def travelling_salesmen():#10
    global auc,data,I,P,Rt
    P[10]=1
    Rt=0
    I_clear()
    I[10]=1
    record()
    total_distance()# i8 i10

    reset_1ptr()  #1 #i11 #i10
    while auc[4]==0:
        shift_to_distance() #i9 i10
    reset_1ptr_4() #1 i14 i10
    auc[3]=data[auc[0]][auc[1]]
    while auc[4]==0:
        if data[auc[0]][auc[1]]<auc[3]:
            auc[3]=data[auc[0]][auc[1]]
        shift_0_1()#1 i16 i10
    reset_2ptr() #2 i12 i10
    while auc[4]==0:
        ptr_to_distance_1()#1 i18 i10
    reset_2ptr_4() #2 i15 i10
    while auc[4]==0:
        if -0.0000001<data[auc[0]][auc[1]]-auc[3]<0.0000001:
            data[auc[0]][8]=1
        shift_0_2() #2 i17 i10
    reset_3ptr() #3 i13 i10
    Rt=1
    P[10]=0
    I_clear()
    record()
def reset_1ptr():#11
    global auc,I,P,Rt,data
    P[11]=1
    I_clear()
    I[11]=1
    Rt=0
    record()
    auc=[0,0,0,0,0]
    P[11]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def reset_2ptr():#12
    global auc,I,P,data,Rt
    P[12]=1
    I_clear()
    I[12]=1
    Rt=0
    record()
    auc[0]=0
    auc[1]=0
    auc[2]=0
    auc[4]=0
    P[12]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def reset_3ptr():#13
    global auc,I,P,data,Rt
    P[13]=1
    I_clear()
    I[13]=1
    Rt=0
    record()
    auc=[0,0,0,0,0]
    P[13]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def reset_1ptr_4():#14
    global auc,data,I,P,Rt
    I_clear()
    I[14]=1
    P[14]=1
    Rt=0
    record()
    auc[4]=0 
    P[14]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def reset_2ptr_4():#15
    global auc,data,I,P,Rt
    P[15]=1
    I_clear()
    I[15]=1
    Rt=0
    record()
    auc[4]=0 
    P[15]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def shift_0_1():#16
    global auc,data,I,P,Rt
    P[16]=1
    I_clear()
    I[16]=1
    Rt=0
    record()
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[16]=0
    I_clear()
    Rt=1
    I[10]=1
    record()
def shift_0_2():#17
    global auc,data,I,P,Rt
    P[17]=1
    I_clear()
    I[17]=1
    Rt=0
    record()
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[17]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def ptr_to_distance_1():#18
    global auc,data,I,P,Rt
    P[18]=18
    I_clear()
    I[18]=1
    Rt=0
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[18]=0
    I_clear()
    I[10]=1
    Rt=1
    record()
def record():
    global auc,data,I,P,Rt,oneroundI,oneroundP,oneroundRt,oneroundauc,onerounddata,counterstep
    oneroundI.append(I)
    oneroundP.append(P)
    oneroundRt.append(Rt)
    oneroundauc.append(auc)
    onerounddata.append(data)
    counterstep+=1
def set_input_output():
    global oneroundI,oneroundP,oneroundRt,oneroundauc,onerounddata,inputdata,outputdataIt_plus_1,outputdataRt,outputdataauct_plus_1
    inputalllist=[]
    inputalllistP=[]
    aRt=[]
    for i in range(len(oneroundI)-1):
        inputlist=[]
        for j in range(120):
            
            for k in range(5):
                inputlist.append(onerounddata[i][j][k][0])
                inputlist.append(onerounddata[i][j][k][1])
            for k in range(4):
                inputlist.append(onerounddata[i][j][k+5])
        for j in range(5):
            inputlist.append(oneroundauc[i][j])
        inputalllistP.append(oneroundP[i])
        inputalllist.append(inputlist)
        aRt.append(oneroundRt[i])
    inputdata.append(inputalllist)
    inputdataP.append(inputalllistP)
    outputdataRt.append(aRt)

    aI=[]
    aauc=[]
    for i in range(len(oneroundI)-1):
        aI.append(oneroundI[i+1])
        aauc.append(oneroundauc[i+1])
    outputdataIt_plus_1.append(aI)
    outputdataauct_plus_1.append(aauc)


inputdata=[]
inputdataP=[]
outputdataIt_plus_1=[]
outputdataRt=[]
outputdataauct_plus_1=[]
for _ in range(100):
    onerounddata=[]
    oneroundauc=[]
    oneroundP=[]
    oneroundRt=[]
    oneroundI=[]
    n = 4
    height =300
    width =300
    Onedata=np.random.randn(n,2)
    data=[]
    counterstep=0
    for i in range(len(Onedata)):
        Onedata[i][0]=Onedata[i][0]*height
        Onedata[i][1]=Onedata[i][1]*width
    for i in range(120):
        data.append([[0,0],[0,0],[0,0],[0,0],[0,0],n,math.factorial(n),0,0])
    k=0
    for i in permutations(Onedata):
        for j in range(len(list(i))):
            data[k][j]=list(i)[j]
        k+=1
    auc=[0,0,0,0,0]
    P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Rt=0
    travelling_salesmen()
    
    while counterstep<1105:
        auc=[0,0,0,0,0]
        P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        Rt=1
        record()
    set_input_output()


    
    
np.savez('traingdata4_100_20.npz',inputdata=inputdata,inputdataP=inputdataP,outputdataIt_plus_1=outputdataIt_plus_1,outputdataRt=outputdataRt,outputdataauct_plus_1=outputdataauct_plus_1)

