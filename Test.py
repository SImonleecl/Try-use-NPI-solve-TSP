import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def shift():
    global data, auc, I, R,traceI,traceP, traceauc,traceenv,traceR
    I=[0,0]
    I[1]=1
    record()
    a=auc[0]
    traceauc.append([a])
    if type(data[-1][auc[0]])!=int:
        auc[0]=auc[0]+1
    else:
        R=R+1
    recordR()
def findposrecord():
    global data, auc, I, R,traceI,traceP, traceauc,traceenv,traceR
    I=[0,0]
    I[0]=1
    record()
    a=auc[0]
    traceauc.append([a])
    while R==0:
        shift()
    recordR()
def record():
    global data, I,traceI,traceP,traceenv
    if type(data[-1][auc[0]])!=int:
        traceenv.append(2)
    else:
        traceenv.append(1)
    traceI.append(I)
    if I==[1,0]:
        traceP.append([1,0])
    elif I==[0,1]:
        traceP.append([1,1])
    else:
        traceP.append([0,0])
    
def recordR():
    global R,traceR
    traceR.append(R)
traceI=[]
traceR=[]
traceauc=[]
traceenv=[]
traceP=[]
data=[]
auc=[0]
I=[0,0]
R=0
traceinput=[]
traceoutput=[]
 
BATCH_START = 0
TIME_STEPS = 22
BATCH_SIZE = 100
INPUT_SIZE = 4
OUTPUT_SIZE = 4
CELL_SIZE = 30
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS,data, auc, I, R,traceI,traceP, traceauc,traceenv,traceR,traceinput,traceoutput
    # xs shape (50batch, 22teps)
    biginput=[]
    bigoutput=[]
    for _ in range(BATCH_SIZE):
        n = 2
        height =1
        width =1
        Onedata=np.random.randn(n,2)
        auc=[0]
        R=0
        traceI=[]
        traceR=[]
        traceauc=[]
        traceenv=[]
        traceP=[]
        traceinput=[]
        traceoutput=[]
        for i in range(len(Onedata)):
            Onedata[i][0]=Onedata[i][0]*height
            Onedata[i][1]=Onedata[i][1]*width

        data.append([])
        for i in range(len(Onedata)):
            data[-1].append([Onedata[i][0],Onedata[i][1]])
        data[-1].append(0)
        data[-1].append(0)
        findposrecord()
        traceI.append([0,0])
        traceP.append([0,0])
        traceR.append(1)
        traceauc.append([0])
        traceenv.append(0)

        for i in range(len(traceI)-1):
            traceinput.append([traceenv[i],traceauc[i][0],traceP[i][0],traceP[i][1]])
        
        while len(traceinput)<22:
            traceinput.append([0,0,0,0])
        for i in range(len(traceI)-1):
            traceoutput.append([traceI[i+1][0],traceI[i+1][1],traceauc[i+1][0],traceR[i]])
        while len(traceoutput)<22:
            traceoutput.append([0,0,0,1])
        biginput.append(traceinput)
        bigoutput.append(traceoutput)
    seq=np.array(biginput)
    res=np.array(bigoutput)
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq, res]

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
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
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps*self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
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
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess,"my_net/save_net.skpt")
    merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter("logs2", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     init = tf.initialize_all_variables()
    # else:
    #     init = tf.global_variables_initializer()
    # sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    for i in range(1):
        seq, res = get_batch()

        feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state
            }


        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)

        np.save('LSTMinput.npy',seq)
        np.save('LSTMoutput.npy',pred)
        print("Saved")
        if i % 20 == 0:
            print(round(cost, 4))
            result = sess.run(merged, feed_dict)
            # writer.add_summary(result, i)
    