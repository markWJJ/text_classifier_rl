# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:09:34 2018

@author: Lenovo
"""
import numpy as np
from random import sample
import pickle as pkl

def pad_batch(inputs, max_sequence_length):

    inputs = [seq[:max_sequence_length] for seq in inputs]
            
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    return inputs_batch_major, sequence_lengths

def get_data(metadata_pkl, max_length):
    metadata = pkl.load(open(metadata_pkl, 'rb'))
    
    word_to_int = metadata['w2idx']
    int_to_word = metadata['idx2w']
    
    sentence_int = metadata['sentence_int']
    label = metadata['label']    
    
    sentence_int, length = pad_batch(sentence_int, max_length)
    
    return word_to_int, int_to_word, sentence_int, label, length

def rand_batch_gen(x, y, length, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield np.array(x)[sample_idx], np.array(y)[sample_idx], np.array(length)[sample_idx]

def get_manual_data(metadata_pkl, max_length):
    metadata = pkl.load(open(metadata_pkl, 'rb'))
        
    sentence_int = metadata['sentence_int']
    label = metadata['label']    
    
    sentence_int, length = pad_batch(sentence_int, max_length)
    
    return sentence_int, label, length