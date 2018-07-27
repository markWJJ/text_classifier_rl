import tensorflow as tf
import numpy as np

# -------------- emb mat--------------
def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_symbol=None, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len], else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        if init_mat is None:
            emb_mat = tf.Variable(tf.random_uniform([dict_size, emb_len], -1.0, 1.0),
                                name="emb_mat",
                                dtype=tf.float32)
            
        elif type(init_mat) == tf.Variable:
            emb_mat = init_mat
            
        else:
            emb_len = init_mat.shape[1]
            extra_symbol_matrix = np.random.uniform(-0.1, 0.1, (len(extra_symbol), emb_len))
            
            emb_mat_ept_and_unk = tf.get_variable("emb_pad_unk", 
                                   [len(extra_symbol), emb_len],  
                                   tf.float32, 
                                   initializer=tf.constant_initializer(extra_symbol_matrix, dtype=tf.float32),
                                   trainable=True)
            
            emb_mat_other = tf.get_variable("emb_mat", 
                                  [dict_size - len(extra_symbol), emb_len], 
                                  tf.float32,
                                  initializer=tf.constant_initializer(init_mat[len(extra_symbol):], 
                                                          dtype=tf.float32),
                                  trainable=True)
            
            emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)
            
        
        return emb_mat