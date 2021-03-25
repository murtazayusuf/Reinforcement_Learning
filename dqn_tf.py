import os
import numpy as np 
import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256, input_dims=(210, 160, 4), chpkt_dir='tmp/dqn'):
        self.lr = lr
        self.name = name
        self.num_actions = num_actions
        self.fcl_dims = fcl_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chpkt_dir, 'deepqnet.ckpt')
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')