# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:21:09 2018

@author: Lenovo
"""

import numpy as np
import pickle

n_classes = 10
sentence_len = 200
noise_len = 10

key_len = 1

word_dict_len = 10
batch_size = 5000

sentence_int = np.zeros((batch_size * n_classes, sentence_len))
key_posistion = np.zeros((batch_size * n_classes, key_len))
label = np.zeros((batch_size * n_classes, n_classes))

for i in range(n_classes):
    for j in range(batch_size):
        sentence_int[i*batch_size + j] = np.random.randint(n_classes, n_classes+noise_len, sentence_len)
        key_pos = np.random.randint(sentence_len, size=key_len)
        sentence_int[i*batch_size + j][key_pos] = i
        key_posistion[i*batch_size + j] = key_pos
        label[i*batch_size + j][i] = 1
        
metadata = {
            'sentence_int' : sentence_int,
            'label' : label,
            'key_pos' : key_posistion
                    }

metadata_pkl = './data/manual_data.pkl'
     
output = open(metadata_pkl, 'wb')
pickle.dump(metadata, output)
output.close()          
        
        
        
        
        
        