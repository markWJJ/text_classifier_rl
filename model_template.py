import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from integration_func import generate_embedding_mat

class ModelTemplate(object):
    __metaclass__ = ABCMeta

    def build_placeholder(self, config):
        
        self.config = config
        self.token_emb_mat = self.config["token_emb_mat"]
        self.vocab_size = int(self.config["vocab_size"])
        self.max_length = int(self.config["max_length"])
        self.emb_size = int(self.config["emb_size"])
        self.extra_symbol = self.config["extra_symbol"]
        self.scope = self.config["scope"]
        self.num_features = int(self.config["num_features"])
        self.num_classes = int(self.config["num_classes"])
        self.ema = self.config.get("ema", False)
        self.grad_clipper = float(self.config.get("grad_clipper", 10.0))
        
        print("--------vocab size---------", self.vocab_size)
        print("--------max length---------", self.max_length)
        print("--------emb size-----------", self.emb_size)
        print("--------extra symbol-------", self.extra_symbol)
        print("--------emb matrix---------", self.token_emb_mat.shape)
        
        # ---- place holder -----
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.sent1_len = tf.placeholder(tf.int32, [None], name='sent1_token_length')
        
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        
        self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        
        self.one_hot_label = tf.one_hot(self.gold_label, 2)
        
        # ------------ other ---------
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
        
        # ---------------- for dynamic learning rate -------------------
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        self.learning_rate_value = float(self.config["learning_rate"])
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.emb_mat = generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                 init_mat=self.token_emb_mat, 
                                 extra_symbol=self.extra_symbol, 
                                 scope='gene_token_emb_mat')
        
        self.s1_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent1_token)  # bs,sl1,tel
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate_updated = False
        # ------ start ------
        self.pred_probs = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.opt = None
        self.train_op = None
        
    @abstractmethod
    def build_network(self):
        pass
    
    @abstractmethod
    def build_loss(self):
        pass

    @abstractmethod
    def build_accuracy(self):
        pass
    
    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(),)
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)
            
    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)
            
    def build_op(self):
        
        self.build_network()
        self.build_loss()
        self.build_accuracy()

        # ------------ema-------------
        if self.ema:
            self.var_ema = tf.train.ExponentialMovingAverage(self.config["var_decay"])
            self.build_var_ema()

        if self.config["mode"] == 'train' and self.ema:
            self.ema = tf.train.ExponentialMovingAverage(self.config["decay"])
            self.build_ema()
            
        # ---------- optimization ---------
        if self.config["optimizer"].lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.config["optimizer"].lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config["optimizer"].lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
            
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        grads_and_vars = self.opt.compute_gradients(self.loss, var_list=trainable_vars)
        
        params = [var for _, var in grads_and_vars]
        gradients = [grad for grad, _ in grads_and_vars]

        grads, _ = tf.clip_by_global_norm(gradients, self.grad_clipper)

        self.train_op = self.opt.apply_gradients(zip(grads, params), global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=100)
        
    def init_step(self, sess):
        assert isinstance(sess, tf.Session)
        sess.run(tf.global_variables_initializer())
        
    def step(self, sess, batch_samples, dropout_keep_prob):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, dropout_keep_prob, 'train')
        [loss, train_op, global_step, accuracy] = sess.run([self.loss, self.train_op, 
                                          self.global_step, self.accuracy], feed_dict=feed_dict)

        return loss, train_op, global_step, accuracy
    
    def infer(self, sess, batch_samples, dropout_keep_prob, mode):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, dropout_keep_prob, mode)
        if mode == "test":
            [loss, logits, pred_probs, accuracy] = sess.run([self.loss, self.logits, 
                                                            self.pred_probs, self.accuracy], 
                                                            feed_dict=feed_dict)
            return loss, logits, pred_probs, accuracy
        elif mode == "infer":
            [logits, pred_probs] = sess.run([self.logits, self.pred_probs], 
                                            feed_dict=feed_dict)
            return logits, pred_probs

    def infer_features(self, sess, batch_samples, dropout_keep_prob, mode):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, dropout_keep_prob, mode)
        
        [matched_representation] = sess.run([self.output_features], 
                                            feed_dict=feed_dict)
        return matched_representation
        