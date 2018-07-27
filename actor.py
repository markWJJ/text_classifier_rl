import tensorflow as tf
import tflearn
import numpy as np
from tensorflow.contrib.rnn import LSTMCell

class ActorNetwork(object):
    """
    action network
    use the state
    sample the action
    """

    def __init__(self, sess, dim, optimizer, learning_rate, embeddings):
        self.global_step = tf.Variable(0, trainable=False, name="ActorStep")
        self.sess = sess
        self.dim = dim
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 10000, 0.95, staircase=True)
        self.init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32)
        
        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.wordvector = embeddings
        
        self.num_other_variables = len(tf.trainable_variables())
        
        self.input_state, self.input_word, self.scaled_out, self.out_state = self.create_actor_network()
        self.network_params = tf.trainable_variables()[self.num_other_variables:]

        self.action_gradient = tf.placeholder(tf.float32, [None, 2])
        self.log_scaled_out = tf.log(self.scaled_out)
        
        self.reward = tf.placeholder(tf.float32, [None, 2])
        
        self.loss = tf.reduce_mean(-self.log_scaled_out * self.reward)
        self.optimize = self.optimizer.minimize(self.loss)
                
    def create_actor_network(self):
        intpu_state_c = tf.placeholder(tf.float32, shape = [None, self.dim], name="cell_state")
        intpu_state_h = tf.placeholder(tf.float32, shape = [None, self.dim], name="cell_state")        
        input_word = tf.placeholder(tf.int32, shape=[None,])
        
        input_w = tf.nn.embedding_lookup(self.wordvector, input_word)
        
        cell = LSTMCell(self.dim, initializer=self.init)
        with tf.variable_scope('Actor/LSTM'):
            out, state1 = cell(input_w, (intpu_state_c, intpu_state_h))
            
        t1 = tflearn.fully_connected(state1.c, 1, name='Actor/FullyConnectedC')
        t2 = tflearn.fully_connected(state1.h, 1, name='Actor/FullyConnectedH')        
        t3 = tflearn.fully_connected(input_w, 1, name='Actor/FullyConnectedWord')

        scaled_out = tflearn.activation(\
                tf.matmul(intpu_state_c, t1.W) + tf.matmul(intpu_state_h, t2.W) \
                + tf.matmul(input_w, t3.W) + t1.b,\
                activation = 'sigmoid')
        
        s_out = tf.clip_by_value(scaled_out, 1e-5, 1 - 1e-5)
        
        scaled_out = tf.concat([1.0 - s_out, s_out], axis=1) 

        input_state = (intpu_state_c, intpu_state_h)
        out_state = (state1.c, state1.h)
        return input_state, input_word, scaled_out, out_state
        
    def train(self, input_state, input_word, reward):
        self.sess.run(self.optimize, feed_dict={
            self.input_state: input_state,
            self.input_word: input_word,
            self.reward: reward})

    def predict_target(self, input_state, input_word):
        return self.sess.run([self.scaled_out, self.out_state], feed_dict={
            self.input_state: input_state,
            self.input_word: input_word})
    
    
    def lower_LSTM_state(self, state, inputs):
        """
        state : (state_c, state_h)
        """
        state_c, state_h = state
        return self.sess.run(self.lower_cell_state1, feed_dict={
            self.lower_cell_state: state,
            self.lower_cell_input: inputs})