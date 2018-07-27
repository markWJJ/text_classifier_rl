import tensorflow as tf
import numpy as np
import pickle as pkl
import os, json

class ModelAPI(object):
    def __init__(self, model_dir, embed_path):
        self.model_dir = model_dir
        self.embed_path = embed_path
        
    def dump_config(self):
        pkl.dump(self.parameter_config, open(os.path.join(self.model_dir, "config.pkl"), "wb"))
        
    def load_config(self):
        self.parameter_config = json.load(open(os.path.join(self.model_dir, "config.json"), "r"))
        embedding_info = pkl.load(open(os.path.join(self.embed_path, "emb_mat.pkl"), "rb"))
        self.config = {}
        self.config["token2id"] = embedding_info["token2id"]
        self.config["id2token"] = embedding_info["id2token"]
        self.config["token_emb_mat"] = embedding_info["embedding_matrix"]
        self.config["id2type"] = embedding_info["id2type"]
        for key in self.config["id2type"].keys():
            print (key, self.config["id2type"][key])
        for key in self.parameter_config:
            self.config[key] = self.parameter_config[key]
        self.batch_size = self.config["batch_size"]

    def update_config(self, updated_config_dict):
        for key in updated_config_dict:
            if key in self.config:
                self.config[key] = updated_config_dict[key]
            else:
                self.config[key] = updated_config_dict[key]
                
    def iter_batch(self, anchor, label, anchor_len, batch_size, mode="train"):

        if isinstance(anchor, list):
            data_num = len(anchor)
        else:
            data_num = anchor.shape[0]

        if mode == "train":
            shuffled_index = np.random.permutation(data_num)
        else:
            shuffled_index = range(data_num)

        if isinstance(anchor, list):
            batch_num = int(data_num / batch_size)
            for t in range(batch_num):
                start_index = t * batch_size
                end_index = start_index + batch_size
                sub_anchor = [anchor[index] for index in shuffled_index[start_index:end_index]]
                sub_label = [label[index] for index in shuffled_index[start_index:end_index]]
                sub_anchor_len = [anchor_len[index] for index in shuffled_index[start_index:end_index]]
                yield sub_anchor, sub_label, sub_anchor_len
            if end_index < data_num:
                sub_anchor = [anchor[index] for index in shuffled_index[end_index:]]
                sub_label = [label[index] for index in shuffled_index[end_index:]]
                sub_anchor_len = [anchor_len[index] for index in shuffled_index[end_index:]]
            
            yield sub_anchor, sub_label, sub_anchor_len

        else:
            batch_num = int(data_num / batch_size)
            for t in range(batch_num):
                start_index = t * batch_size
                end_index = start_index + batch_size
                sub_anchor = anchor[shuffled_index[start_index:end_index]]
                sub_label = label[shuffled_index[start_index:end_index]]
                sub_anchor_len = anchor_len[shuffled_index[start_index:end_index]]
                yield sub_anchor, sub_label, sub_anchor_len
            if end_index < data_num:
                sub_anchor = anchor[shuffled_index[end_index:]]
                sub_label = label[shuffled_index[end_index:]]
                sub_anchor_len = anchor_len[shuffled_index[end_index:]]
            
            yield sub_anchor, sub_label, sub_anchor_len
        
    def build_graph(self, sess, model, device="/cpu:0"):
#        self.graph = tf.Graph()
        self.graph = tf.get_default_graph()
        with self.graph.as_default() as g:
            g.device(device)
            session_conf = tf.ConfigProto(
              intra_op_parallelism_threads=4, # control inner op parallelism threads
              inter_op_parallelism_threads=4, # controal among op parallelism threads
              device_count={'CPU': 4, 'GPU': 1},
              allow_soft_placement=True,
              log_device_placement=False)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config["gpu_ratio"]) 
#            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess = sess
            self.model = model
            self.model.build_placeholder(self.config)
            print("-------Succeeded in building placeholder---------")
            self.model.build_model()
            print("-------Succeeded in building model-------")
            self.model.init_step(self.sess)
            print("-------Succeeded in initializing model-------")
     
    def load_model(self, load_type):
        if load_type == "specific":
            model_path = os.path.join(self.model_dir, self.model_type+".ckpt")
            self.model.saver.restore(self.sess, model_path)
        elif load_type == "latest":
            print(tf.train.latest_checkpoint(self.model_dir))
            self.model.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        print("-----------succeeded in restoring pretrained model------------")
                
    def infer_step(self, sample_batch):
        [logits, pred_probs] = self.model.infer(self.sess, sample_batch, self.config["dropout_keep_prob"], "infer")
        return logits, pred_probs
           
    def train_step(self, train_dataset, dev_dataset):
        self.best_dev_accuracy = -100.0
        self.early_stop_step = self.config["early_stop_step"]
        [train_anchor, train_label, train_anchor_len] = train_dataset
        [dev_anchor, dev_label, dev_anchor_len] = dev_dataset

        train_cnt = 0
        stop_step = 0
        stop_flag = False
        for epoch in range(self.config["max_epoch"]):
            train_batch = self.iter_batch(train_anchor, train_label, 
                               train_anchor_len, self.batch_size)
            
            train_loss = 0.0
            train_accuracy = 0.0
            train_internal_cnt = 0
            for train in train_batch:

                [sub_anchor, 
                sub_label, 
                sub_anchor_len] = train

                sub_anchor = np.array(sub_anchor).astype(np.int32)
                sub_label = np.array(sub_label).astype(np.int32)
#                sub_anchor_len = 0
                
                [loss, train_op, 
                 global_step, accuracy] = self.model.step(self.sess, [sub_anchor, sub_label, 
                                                  sub_anchor_len], self.config["dropout_keep_prob"])
                
                print('loss:',loss,'acc:',accuracy)
                train_cnt += 1
                train_internal_cnt += 1
                train_loss += loss
                train_accuracy += accuracy
                
                if np.mod(train_cnt, self.config["validation_step"]) == 0:
                    dev_cnt = 0
                    dev_accuracy = 0.0
                    dev_batch = self.iter_batch(dev_anchor, dev_label, 
                               dev_anchor_len, self.batch_size, "dev")
                    for dev in dev_batch:
                        [sub_anchor, 
                        sub_label, 
                        sub_anchor_len] = dev

                        sub_anchor = np.array(sub_anchor).astype(np.int32)
                        sub_label = np.array(sub_label).astype(np.int32)
                        sub_anchor_len = 0

                        [loss, logits, pred_probs, accuracy] = self.model.infer(self.sess, 
                                                    [sub_anchor, sub_label, 
                                                  sub_anchor_len],                                                                                   self.config["dropout_keep_prob"], "test")
                        dev_cnt += 1
                        dev_accuracy += accuracy
                    dev_accuracy /= float(dev_cnt)
                    if dev_accuracy > self.best_dev_accuracy:
                        self.best_dev_accuracy = dev_accuracy
                        self.model.saver.save(self.sess, 
                                       os.path.join(self.model_dir, self.config["model_type"]+".ckpt"), 
                                       global_step=global_step)
                        stop_step = 0
                        print("-----------succeeded in storing model---------")
                    else:
                        if epoch >= 20:
                            stop_step += 1
                            if stop_step > self.early_stop_step:
                                print("-------------accuracy of development-----------", dev_accuracy, self.best_dev_accuracy)
                                stop_flag = True
                                break
                    print("-------------accuracy of development-----------", dev_accuracy, self.best_dev_accuracy)
            print("--------accuracy of training----------", train_loss/float(train_internal_cnt), train_accuracy/float(train_internal_cnt))
                            
            if stop_step > self.early_stop_step and stop_flag == True:
                break
                        
    def batch_loss(self, anchor, label):
        feed_dict = {
                self.model.sent1_token: anchor,
                self.model.gold_label: label,
                self.model.is_train: False,
                self.model.dropout_keep_prob: self.config["dropout_keep_prob"]
                }
        loss = self.sess.run(self.model.batch_loss, feed_dict=feed_dict)
        return loss
                
    def train_batch(self, anchor, label):   
        feed_dict = {
                self.model.sent1_token: anchor,
                self.model.gold_label: label,
                self.model.is_train: True,
                self.model.learning_rate: self.model.learning_rate_value,
                self.model.dropout_keep_prob: self.config["dropout_keep_prob"]
                }
        self.sess.run(self.model.train_op, feed_dict=feed_dict)
    
    def predict_batch(self, anchor):
        feed_dict = {
                self.model.sent1_token: anchor,
                self.model.is_train: False,
                self.model.dropout_keep_prob: 1
                }
        
        pred_probs = self.sess.run(self.model.pred_probs, 
                                   feed_dict=feed_dict)
        return pred_probs

