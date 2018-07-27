
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:14:38 2018

@author: Lenovo
"""

import numpy as np
import tensorflow as tf
import random
import os
import json
import pickle as pkl

from actor import ActorNetwork
from self_att_rnn import SelfAttRNN
from model_api import ModelAPI
from integration_func import generate_embedding_mat
from data_helper import get_manual_data, rand_batch_gen


metadata_pkl = './data/manual_data.pkl'
max_length = 200

sentence_int, label, length = get_manual_data(metadata_pkl, max_length)
rand_batch = rand_batch_gen(sentence_int, label, length, 64)  
vocab_size = 20
embed_size = 16

options = {
    "context_lstm_dims":100,
    "highway_layer_num":1
}

config = {
            "options":options,
            "vocab_size":vocab_size,
            "max_length":max_length,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"self_att_rnn",
            "num_features":100,
            "num_classes":10,
            "filter_width":4,
            "di":50,
            "grad_clipper":10.0,
            "l2_reg":1e-5,
            "model_type":"SelfAttRNN",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-4,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "loss_type":"cross_entropy",
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":5,
            "validation_step":10000,
            "gpu_ratio":0.9,
            "filter_sizes":[3,4,5],
            "num_filters":128,
            "wd":5e-5
}

model_dir = './model/'
embed_path = './data/'
json.dump(config, open(model_dir + "config.json", "w"))

lstm_dim = 64
epsilon = 0.2
alpha = 0.1
lr = 0.0005
batchsize = 64
samplecnt = 50

embedding_size = 32
model_name = 'test'

dropout = config["dropout_keep_prob"]
optimizer = config["optimizer"]
num_classes = config["num_classes"]

    
def sampling_RL(sess, actor, input_vec, lenth, epsilon=0., Random=True):
    n_batch = len(input_vec)
    
    current_lower_state_c = np.zeros((n_batch, lstm_dim), dtype=np.float32)
    current_lower_state_h = np.zeros((n_batch, lstm_dim), dtype=np.float32)
    current_lower_state = (current_lower_state_c, current_lower_state_h)    
    
    states = []
    actions = []
    
    #sampling actions   
    for pos in range(max(lenth)):
        predicted, state = actor.predict_target(current_lower_state, input_vec[:, pos])
        
        states.append([current_lower_state, input_vec[:, pos]])
        if Random:
            random_matrix = np.random.rand(n_batch)
            action = np.ones(n_batch)
            if random.random() > epsilon:
                action[random_matrix < predicted[:, 0]] = 0
            else:
                action[random_matrix > predicted[:, 0]] = 0
        else:
            action = np.argmax(predicted, axis=1)
        actions.append(action)
        
        state_c, state_h = state
        
        current_upper_state_c = np.zeros((n_batch, lstm_dim), dtype=np.float32)
        current_upper_state_h = np.zeros((n_batch, lstm_dim), dtype=np.float32)
        for i in range(n_batch):
            current_upper_state_c[i] = state_c[i]*action[i] + (1-action[i])*current_lower_state_c[i]
            current_upper_state_h[i] = state_h[i]*action[i] + (1-action[i])*current_lower_state_h[i]
        
        current_lower_state = (current_upper_state_c, current_upper_state_h)
             
    actions = np.array(actions).astype('int32')
    a = np.zeros((len(states), n_batch))
    for i in range(len(states)):
        a[i] = input_vec[:, i] * actions[i] 
    
    Rinput = a.transpose()
    word_index = (Rinput != 0)
    Rlenth = np.sum(word_index, axis=1).astype('int32')
    
    Rinput_pad = np.zeros(Rinput.shape)

    for i in range(len(Rinput)):
        Rinput_pad[i][:Rlenth[i]] = Rinput[i][word_index[i]]
    
    Rinput = Rinput_pad
    return actions, states, Rinput, Rlenth


def train_classifier(sess, classifier):
    for i in range(1000):
    
        input_vec, onehot_label, lenth = next(rand_batch)
        batch_label = np.argmax(onehot_label, axis=1)
        classifier.train_batch(input_vec, batch_label)
        
        if i % 10 == 0:   
            acc_test, loss  = test_classifier(sess, classifier)   
            print ("batch", i, "--accuracy: {:.2f}%".format(acc_test*100))
            print ("\t", "--batch loss: {:.2f}".format(loss))

def test_classifier(sess, classifier):
    rand_batch = rand_batch_gen(sentence_int, label, length, 640)  
    input_vec, onehot_label, lenth = next(rand_batch)    
    batch_label = np.argmax(onehot_label, axis=1)
        
    out = classifier.predict_batch(input_vec)
    loss = classifier.batch_loss(input_vec, batch_label)
    
    correct = np.argmax(out, axis=1) == batch_label
    acc = np.sum(correct) / 640
    ave_loss = np.sum(loss) / 640

    return acc, ave_loss                

def test(sess, actor, classifier, RL=True):
    
    rand_batch = rand_batch_gen(sentence_int, label, length, 640)  
    input_vec, onehot_label, lenth = next(rand_batch)
    
    batch_label = np.argmax(onehot_label, axis=1)
    
    if RL:
        actions, states, Rinput, Rlenth = sampling_RL(sess, actor, input_vec, lenth, epsilon, Random=False)        
        out = classifier.predict_batch(Rinput)
        loss = classifier.batch_loss(Rinput, batch_label)
         
        for i in range(5):
            print('\nexample:')
            input_i = i
            print(input_vec[input_i])
            print('label: ', np.min(input_vec[input_i]))
            print('\nafter Simplifying:')
            print(Rinput[input_i])
            print()
            
        keep_ratio = np.mean(Rlenth / lenth)
        print('keep ratio: {:.2f}%'.format(keep_ratio*100))  
    else:
        out = classifier.predict_batch(input_vec, lenth)
        loss = classifier.batch_loss(input_vec, lenth, batch_label)
        
    correct = (np.argmax(out, axis=1) == np.argmax(onehot_label, axis=1))
    acc = np.sum(correct) / 640
    ave_loss = np.sum(loss) / 640

    return acc, ave_loss

def train(sess, actor, classifier, batchsize, classifier_trainable=True):

    for b in range(1000):           
        acc_test, loss = test(sess, actor, classifier)           
        print ("batch", b, " --batch accuracy: {:.2f}%".format(acc_test*100))
        print ("\t", "--batch loss: {:.2f}".format(loss))       
        
        rand_batch = rand_batch_gen(sentence_int, label, length, batchsize)  
        input_vec, onehot_label, lenth = next(rand_batch)
        
        batch_label = np.argmax(onehot_label, axis=1)
        
        actionlist, statelist, losslist = [], [], []
        for i in range(samplecnt):
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, input_vec, lenth, epsilon, Random=True)
            actionlist.append(actions)
            statelist.append(states)
            
            loss = classifier.batch_loss(Rinput, batch_label) 
            length_loss = np.maximum(Rlenth-20, 0)**2 / (80)**2 * 0.2
#            length_loss = (Rlenth / lenth) ** 2 * 0.2
            loss += length_loss
            
            losslist.append(loss)
            
            if i % 20 == 0:
                print('lenth loss:{:.2f}'.format(np.mean(length_loss)))
                keep_ratio = np.mean(Rlenth / lenth)
                print('random keep ratio: {:.2f}%'.format(keep_ratio*100))   

            if classifier_trainable:
                if i % 20 == 0:
                    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, input_vec, lenth, epsilon, Random=False)
                    classifier.train_batch(Rinput, batch_label)
                    
                    keep_ratio = np.mean(Rlenth / lenth)
                    print('argmax keep ratio: {:.2f}%'.format(keep_ratio*100))   

        loss_array = np.array(losslist)
        aveloss = np.mean(loss_array, axis=0)
                
                
        for i in range(samplecnt):
            reward = -(losslist[i] - aveloss)

            actions = actionlist[i]
            states = statelist[i]
            
            total_time = len(states)
            batchsize = len(reward)               
            for time in range(total_time): 
                rr = np.zeros((batchsize, 2)) 
                action = actions[time]

                for i in range(batchsize):
                    rr[i, action[i]] = reward[i]
                
                actor.train(states[time][0], states[time][1], rr)


def restore_model(sess, saver, model_file):
    saver = tf.train.Saver()
    if os.path.exists(model_file + '.meta'):
        saver.restore(sess, model_file)
        print('restore from {}.meta'.format(model_file))
    else:
        sess.run(tf.global_variables_initializer())

def get_model_params(sess):
    gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, sess.run(gvars))}

def check_result(sess, actor):
    '''
    check retained ratio of sentence
    '''
    input_vec, solution, lenth = next(rand_batch)
    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, input_vec, lenth, epsilon, Random=False)
    keep_ratio = np.mean(Rlenth / lenth)

    print('keep ratio: {:.2f}%'.format(keep_ratio*100))      


    
def get_simplify(sess, actor):
    rand_batch = rand_batch_gen(sentence_int, label, length, 640)  
    input_vec, onehot_label, lenth = next(rand_batch)
    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, input_vec, lenth, epsilon, Random=False)        
        
    text = ''
    for i in range(len(input_vec)):
        text += 'example: \n'
        
        text += 'after simplifying: \n'        
        simplify = Rinput
        text += ''.join(simplify) + '\n\n\n'
        
    with open('./data/example.txt', 'w') as f:
        f.write(text)
    
    
class_list = list(range(10))

word_to_int = {i:i for i in range(20)}
int_to_word = {i:i for i in range(20)}
def save_embedding_info(embed_file):
        
    type2id = {type_ : id_ for id_, type_ in enumerate(class_list)}
     
    pkl.dump({"token2id":word_to_int,
           "id2token":int_to_word,
           "embedding_matrix":None,
           "id2type":type2id}, 
           open(embed_file, "wb"))
    
def main():
    tf.reset_default_graph()
             
    with tf.Session() as sess:
    
        embeddings = generate_embedding_mat(vocab_size, embedding_size)
                      
        embed_file = './data/emb_mat.pkl'
        if not os.path.exists(embed_file):
            save_embedding_info(embed_file)
            
        classifier = ModelAPI(model_dir, embed_path)
        classifier.load_config()
        classifier.config["token_emb_mat"] = embeddings    
        model = SelfAttRNN()
        classifier.build_graph(sess, model, "/gpu:2")   
        
        actor = ActorNetwork(sess, lstm_dim, optimizer, lr, embeddings)
        
        saver = tf.train.Saver()    
        model_file = "./checkpoints/{}".format(model_name)
        restore_model(sess, saver, model_file)
#        params = get_model_params(sess)
#        get_simplify(sess, actor)
        
        epoch = 5
        try:
            for e in range(epoch):
                if use_RL:
                    train(sess, actor, classifier, batchsize, classifier_trainable=True)
                else:
                    train_classifier(sess, classifier)

                saver.save(sess, model_file)
        
        except KeyboardInterrupt:
                print('[INFO] Interrupt manually, try saving checkpoint for now...')
                saver.save(sess, model_file)

if __name__ == "__main__":
    use_RL = False
    main()
    

