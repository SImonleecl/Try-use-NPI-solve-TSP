import numpy as np 

def NPI_loop(key):
    global R ,outputdata,data,result
    tempkey = key
    P=[data[tempkey][2],data[tempkey][3]]
    KeyA=key
    
    while R <0.5:
        R=outputdata[KeyA][3]
        rt=R
        I2=[outputdata[KeyA][0],outputdata[KeyA][1]]
        P2=I_to_P(I2)
        if ACT(P):
            shift(KeyA)
            KeyA+=1
        else:
            NPI_loop(KeyA+1)

def ACT(lst):
    if lst==[1,1]:
        return True 
    else:
        return False
def shift(Key):
    global data,result 
    if data[Key][0]==2:
        result=data[Key][1]+1
def I_to_P(lstI):
    tempP=[0,0]
    if lstI == [1,0]:
        tempP=[1,0]
    elif lstI == [0,1]:
        tempP=[1,1]
    return tempP
indata=np.load('LSTMinput.npy')
indata=np.reshape(indata,[100,22,4])
R=0
output=np.load('LSTMoutput.npy')
output=np.reshape(output.astype(int),[100,22,4])
result=0
print(np.shape(indata))
print(np.shape(output))
# print(indata)
# print(output)
passnum=0
correctnum=0
for i in range(100):
    result=0
    R=0
    data=indata[i]
    outputdata=output[i]
    try:
        NPI_loop(0)
        passnum+=1
        
    except Exception:
        pass
    if result==6:
        correctnum+=1
    
print("total testnumber: 100, pass number: {}, correct number:{}".format(passnum,correctnum))