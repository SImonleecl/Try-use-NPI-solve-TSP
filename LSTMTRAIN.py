import numpy as np
import math
from itertools import permutations
import tensorflow as tf 

def I_clear():
    global I
    I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def reset_ptr():#0
    global auc,I,P,data,Rt
    I_clear()
    I[0]=1
    P[0]=1
    record()
    auc=[0,0,0,0,0]
    P[0]=0
    Rt=1
    recordRt()
def ptr_to_distance():#1
    global auc,data,I,P,Rt
    P[1]=1
    I_clear()
    I[1]=1
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[1]=0
    Rt=auc[4]
    recordRt()

def reset_ptr_4():#2
    global auc,data,I,P,Rt
    P[2]=1
    I_clear()
    I[2]=1
    record()
    auc[4]=0 
    P[2]=0
    Rt=1
    recordRt()

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
    auc[2]+=1
    P[3]=0
    Rt=auc[4]
    recordRt()

def reset_ptr_1to4():#4
    global auc,data,I,P,Rt
    P[4]=1
    I_clear()
    I[4]=1
    record()
    auc[2]=0
    auc[3]=0
    auc[4]=0
    auc[1]=0
    P[4]=0
    Rt=1
    recordRt()

def linedistance():#5
    global auc,data,I,P,Rt
    P[5]=1
    I_clear()
    I[5]=1
    record()
    while auc[4]==0:
        ptr_to_distance()#i1#i6
    reset_ptr_4()#i2i6
    while auc[4]==0:
        twopointdistance()#i3#i6
    reset_ptr_1to4()#i5i6
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[5]=0
    Rt=auc[4]
    recordRt()

def total_distance():#6
    global auc,data,I,P,Rt #i8
    P[6]=1
    I_clear()
    I[6]=1
    record()
    reset_ptr() #i0 #i8
    if auc[0]<data[0][6]:
        while auc[4]==0:
            linedistance()#i6 #i8
    P[6]=0
    Rt=1
    recordRt()

def shift_to_distance():#7
    global auc,data,I,P,Rt
    P[7]=1
    I_clear()
    I[7]=1
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[7]=0
    Rt=auc[4]
    recordRt()

def travelling_salesmen():#8
    global auc,data,I,P,Rt
    P[8]=1
    I_clear()
    I[8]=1
    record()
    total_distance()# i8 i10

    reset_1ptr()  #1 #i11 #i10
    while auc[4]==0:
        shift_to_distance() #i9 i10
    reset_1ptr_4() #1 i14 i10
    assign_auc3()
    while auc[4]==0:
        findmin()
    reset_2ptr() #2 i12 i10
    while auc[4]==0:
        ptr_to_distance_1()#1 i18 i10
    reset_2ptr_4() #2 i15 i10
    while auc[4]==0:
        outputsetting() #2 i17 i10
    reset_3ptr() #3 i13 i10
    Rt=1
    P[8]=0
    recordRt()
def assign_auc3():#9
    global auc,I,P,Rt,data
    P[9]=1
    I_clear()
    I[9]=1
    record()
    auc[3]=data[auc[0]][auc[1]]
    Rt=1
    P[9]=0
    recordRt()

def findmin():#10
    global auc,data,I,P,Rt
    P[10]=1
    I_clear()
    I[10]=1
    record()
    if data[auc[0]][auc[1]]<auc[3]:
        auc[3]=data[auc[0]][auc[1]]
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[10]=0
    Rt=auc[4]
    recordRt()

def outputsetting():#11
    global auc,data,I,P,Rt
    P[11]=1
    I_clear()
    I[11]=1
    record()
    if -0.0000001<data[auc[0]][auc[1]]-auc[3]<0.0000001:
        data[auc[0]][8]=1
    if auc[0]+1<data[0][6]:
        auc[0]+=1
    else:
        auc[4]+=1
    P[11]=0
    Rt=auc[4]
    recordRt()

def reset_1ptr():#12
    global auc,I,P,Rt,data
    P[12]=1
    I_clear()
    I[12]=1
    record()
    auc=[0,0,0,0,0]
    P[12]=0
    Rt=1
    recordRt()

def reset_2ptr():#13
    global auc,I,P,data,Rt
    P[13]=1
    I_clear()
    I[13]=1
    record()
    auc[0]=0
    auc[1]=0
    auc[2]=0
    auc[4]=0
    P[13]=0
    Rt=1
    recordRt()
 
def reset_3ptr():#14
    global auc,I,P,data,Rt
    P[14]=1
    I_clear()
    I[14]=1
    record()
    auc=[0,0,0,0,0]
    P[14]=0
    Rt=1
    recordRt()

def reset_1ptr_4():#15
    global auc,data,I,P,Rt
    I_clear()
    I[15]=1
    P[15]=1
    record()
    auc[4]=0 
    P[15]=0
    Rt=1
    recordRt()

def reset_2ptr_4():#16
    global auc,data,I,P,Rt
    P[16]=1
    I_clear()
    I[16]=1
    record()
    auc[4]=0 
    P[16]=0
    Rt=1
    recordRt()

def ptr_to_distance_1():#17
    global auc,data,I,P,Rt
    P[17]=17
    I_clear()
    I[17]=1
    record()
    if 7>auc[1]:
        auc[1]+=1
    else:
        auc[4]+=1
    P[17]=0
    Rt=auc[4]
    recordRt()

def record():
    global auc,data,I,P,oneroundI,oneroundP,oneroundauc,onerounddata,counterstep
    oneroundI.append(I)
    oneroundP.append(P)
    oneroundauc.append(auc)
    onerounddata.append(data)
    counterstep+=1
def recordRt():
    global Rt,oneroundRt
    oneroundRt.append(Rt)
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

########################################################################################
inputdata=[]
inputdataP=[]
outputdataIt_plus_1=[]
outputdataRt=[]
outputdataauct_plus_1=[]
onerounddata=[]
oneroundauc=[]
oneroundP=[]
oneroundRt=[]
oneroundI=[]
auc=[0,0,0,0,0]
P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Rt=0
data=[]
ecow1=np.load('ecow1.npy')
ecow2=np.load('ecow2.npy')
ecob1=np.load('ecob1.npy')
ecob2=np.load('ecob2.npy')

BATCH_START = 0
TIME_STEPS = 434
BATCH_SIZE = 2
INPUT_SIZE = 418
OUTPUT_SIZE = 50
OUTPUT_SIZEI = 19
OUTPUT_SIZEauc=5
OUTPUT_SIZErt=1
CELL_SIZE = 500
LR = 0.006
counterstep=0

def get_batch():
    global data,auc,P,I,Rt,ecob1,ecob2,ecow1,ecow2,TIME_STEPS,BATCH_SIZE,\
    inputdata,inputdataP,outputdataIt_plus_1,outputdataRt,outputdataauct_plus_1,\
    onerounddata,oneroundauc,oneroundP,oneroundRt,oneroundI
    # xs shape (50batch, 20steps)
    inputdata=[]
    inputdataP=[]
    outputdataIt_plus_1=[]
    outputdataRt=[]
    outputdataauct_plus_1=[]
    for _ in range(200):
        onerounddata=[]
        oneroundauc=[]
        oneroundP=[]
        oneroundRt=[]
        oneroundI=[]
        n = 3
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
        P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        Rt=0
        travelling_salesmen()
        while counterstep<435:
            auc=[0,0,0,0,0]
            P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            I=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
            Rt=1
            record()
            recordRt()
        set_input_output()
    indata=np.array(inputdata).reshape([-1,1685])
    eco_layer1=np.add(np.matmul(indata,ecow1),ecob1)
    state_t=np.add(np.matmul(eco_layer1,ecow2),ecob2).reshape([BATCH_SIZE,-1,400])
    inP=np.array(inputdataP)
    xs=np.append(state_t,inP,axis=1)#400+18
    outIt_p_1=np.array(outputdataIt_plus_1)
    outacu_p_1=np.array(outputdataauct_plus_1)
    outRt=np.array(outputdataRt)
    # returned seq, res and xs: shape (batch, step, input)
    print("shape1: ",np.shape(outIt_p_1))
    print("shape2:",np.shape(outIt_p_1[:, :, :]))
    return [outIt_p_1[:, :, :], outRt[:, :, :], outacu_p_1[:, :, :],xs]


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size,output_size1,output_size2,output_size3, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.output_size1 = output_size1
        self.output_size2= output_size2
        self.output_size3 = output_size3
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.Its = tf.placeholder(tf.float32, [None, n_steps, output_size1], name='Its')
            # self.aucs = tf.placeholder(tf.float32, [None, n_steps, output_size2], name='aucs')
            # self.Rts = tf.placeholder(tf.float32, [None, n_steps, output_size3], name='Rts')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):

        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
         # shape = (batch * steps, cell_size)
        
        l_to_It = tf.reshape(self.pred, [-1, self.output_size], name='2_2D')
        Ws_out1 = self._weight_variable([self.output_size, self.output_size1])
        bs_out1 = self._bias_variable([self.output_size1, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b1'):
            self.predIt = tf.nn.sigmoid(tf.matmul(l_to_It, Ws_out1) + bs_out1)
        #  # shape = (batch * steps, cell_size)
        # l_to_auc = tf.reshape(self.pred, [-1, self.output_size], name='2_2D')
        # Ws_out2 = self._weight_variable([self.output_size, self.output_size2])
        # bs_out2 = self._bias_variable([self.output_size2, ])
        # # shape = (batch * steps, output_size)
        # with tf.name_scope('Wx_plus_b2'):
        #     self.predauc = tf.nn.relu(tf.matmul(l_to_auc, Ws_out2) + bs_out2)
        #  # shape = (batch * steps, cell_size)
        # l_to_Rt = tf.reshape(self.pred, [-1, self.output_size], name='2_2D')
        # Ws_out3 = self._weight_variable([self.output_size, self.output_size3])
        # bs_out3 = self._bias_variable([self.output_size3, ])
        # # shape = (batch * steps, output_size)
        # with tf.name_scope('Wx_plus_b3'):
        #     self.predRt = tf.nn.sigmoid(tf.matmul(l_to_Rt, Ws_out3) + bs_out3)
        lossesIt = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.predIt, [-1], name='reshape_predIt')],
            [tf.reshape(self.Its, [-1], name='reshape_targetIt')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='lossesIt'
        )
        # lossesauc = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [tf.reshape(self.predauc, [-1], name='reshape_predauc')],
        #     [tf.reshape(self.aucs, [-1], name='reshape_targetauc')],
        #     [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='lossesauc'
        # )
        # lossesRt = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [tf.reshape(self.predRt, [-1], name='reshape_predRt')],
        #     [tf.reshape(self.Rts, [-1], name='reshape_targetRt')],
        #     [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='lossesRt'
        # )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(lossesIt, name='lossesIt_sum'),
                tf.multiply(self.batch_size,self.output_size1),
                name='average_cost')
            self.cost = self.cost+tf.div(
                tf.reduce_sum(lossesauc, name='lossesauc_sum'),
                tf.multiply(self.batch_size,self.output_size2),
                name='average_cost')
            self.cost = self.cost+tf.div(
                tf.reduce_sum(lossesRt, name='lossesRt_sum'),
                tf.multiply(self.batch_size,self.output_size3),
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    get_batch()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, OUTPUT_SIZEI,OUTPUT_SIZEauc,OUTPUT_SIZErt,CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    for i in range(200):
        It_p_1,R_t,auc_p_1, xs = get_batch()
        if i == 0:
            feed_dict = {
                    model.xs: xs,
                    model.Its: It_p_1,
                    # model.aucs:auc_p_1,
                    # model.Rts:R_t
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: xs,
                model.Its: It_p_1,
                # model.aucs:auc_p_1,
                # model.Rts:R_t,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
    


