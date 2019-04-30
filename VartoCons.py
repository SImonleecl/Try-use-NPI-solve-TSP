from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
n_input = 1685  # MNIST data input (img shape: 28*28)

# hidden layer settings
n_hidden_1 = 800# 1st layer num features
n_hidden_2 = 400 # 2nd layer num features

Eco_w1=tf.Variable(np.zeros([n_input, n_hidden_1]),name='ecoh1w', dtype=tf.float32)
Eco_w2=tf.Variable(np.zeros([n_hidden_1, n_hidden_2]),name='ecoh2w', dtype=tf.float32)
Eco_b1=tf.Variable(np.zeros([n_hidden_1]),name='ecob1w', dtype=tf.float32)
Eco_b2=tf.Variable(np.zeros([n_hidden_2]),name='ecob2w', dtype=tf.float32)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_encoder/save_net.ckpt")
    
    ecow1=sess.run(Eco_w1)
    ecow2=sess.run(Eco_w2)
    ecob1=sess.run(Eco_b1)
    ecob2=sess.run(Eco_b2)
    np.save('ecow1.npy',ecow1)
    print(np.shape(ecow1))
    np.save('ecow2.npy',ecow2)
    np.save('ecob1.npy',ecob1)
    np.save('ecob2.npy',ecob2)
    print("Done!")
