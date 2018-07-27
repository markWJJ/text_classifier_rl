"""
including embedding, rnn encoder, hierarchical encoder, attention mechnism
lstm encoder and conv encoder
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from rnn_utils import last_relevant_output

VERY_LARGE_NEGATIVE_VALUE = -1e12
VERY_SMALL_POSITIVE_VALUE = 1e-12

VERY_LARGE_NEGATIVE_VALUE = -1e12
VERY_SMALL_POSITIVE_VALUE = 1e-12

def exp_mask(logits, mask, mask_is_length=True):
	"""Exponential mask for logits.

	  Logits cannot be masked with 0 (i.e. multiplying boolean mask)
	  because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
	  to `False` portion of `mask` so that the portion is effectively ignored
	  when exponentiated, e.g. softmaxed.

	Args:
		logits: Arbitrary-rank logits tensor to be masked.
		mask: `boolean` type mask tensor.
		  Could be same shape as logits (`mask_is_length=False`)
		  or could be length tensor of the logits (`mask_is_length=True`).
		mask_is_length: `bool` value. whether `mask` is boolean mask.
	Returns:
		Masked logits with the same shape of `logits`.
	"""
	if mask_is_length:
		mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
	return logits + (1.0 - tf.cast(mask, 'float')) * VERY_LARGE_NEGATIVE_VALUE

def bi_rnn(hidden_size,
		   inputs_list,
		   sequence_length_list=None,
		   scope=None,
		   dropout_rate=0.0,
		   training=False,
		   stack=False,
		   cells=None,
		   postprocess='concat',
		   return_outputs=True,
		   out_dim=None,
		   reuse=False):
	"""Bidirectional RNN with `BasicLSTMCell`.

	Args:
		hidden_size: `int` value, the hidden state size of the LSTM.
		inputs_list: A list of `inputs` tensors, where each `inputs` is
			single sequence tensor with shape [batch_size, seq_len, hidden_size].
			Can be single element instead of list.
		sequence_length_list: A list of `sequence_length` tensors.
			The size of the list should equal to that of `inputs_list`.
		Can be a single element instead of a list.
		scope: `str` value, variable scope for this function.
		dropout_rate: `float` value, dropout rate of LSTM, applied at the inputs.
		training: `bool` value, whether current run is training.
		stack: `bool` value, whether to stack instead of simultaneous bi-LSTM.
		cells: two `RNNCell` instances. If provided, `hidden_size` is ignored.
		postprocess: `str` value: `raw` or `concat` or `add`.
		Postprocessing on forward and backward outputs of LSTM.
		return_outputs: `bool` value, whether to return sequence outputs.
			Otherwise, return the last state.
		out_dim: `bool` value. If `postprocess` is `linear, then this indicates
			the output dim of the linearity.
		reuse: `bool` value, whether to reuse variables.
	Returns:
		A list `return_list` where each element corresponds to each element of
		input_list`. If the `input_list` is a tensor, also returns a tensor.
	Raises:
		ValueError: If argument `postprocess` is an invalid value.
	"""
	if not isinstance(inputs_list, list):
		inputs_list = [inputs_list]
	if sequence_length_list is None:
		sequence_length_list = [None] * len(inputs_list)
	elif not isinstance(sequence_length_list, list):
		sequence_length_list = [sequence_length_list]
	assert len(inputs_list) == len(sequence_length_list), '`inputs_list` and `sequence_length_list` must have same lengths.'
	with tf.variable_scope(scope or 'bi_rnn', reuse=reuse) as vs:
		if cells is not None:
			cell_fw = cells[0]
			cell_bw = cells[1]
		else:
			cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=reuse)
			cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=reuse)
		return_list = []
		for inputs, sequence_length in zip(inputs_list, sequence_length_list):
			if return_list:
				vs.reuse_variables()
			inputs = tf.nn.dropout(inputs, dropout_rate)
			if stack:
				o_bw, state_bw = tf.nn.dynamic_rnn(
									cell_bw,
									tf.reverse_sequence(inputs, sequence_length, seq_dim=1),
									sequence_length=sequence_length,
									dtype='float',
									scope='rnn_bw')
				o_bw = tf.reverse_sequence(o_bw, sequence_length, seq_dim=1)
				o_bw = tf.nn.dropout(o_bw, dropout_rate)
				o_fw, state_fw = tf.nn.dynamic_rnn(
									cell_fw,
									o_bw,
									sequence_length=sequence_length,
									dtype='float',
									scope='rnn_fw')
			else:
				(o_fw, o_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
									cell_fw,
									cell_bw,
									inputs,
									sequence_length=sequence_length,
									dtype='float')
			return_fw = o_fw if return_outputs else state_fw[-1]
			return_bw = o_bw if return_outputs else state_bw[-1]
			if postprocess == 'raw':
				return_ = return_fw, return_bw
			elif postprocess == 'concat':
				return_ = tf.concat([return_fw, return_bw], 2 if return_outputs else 1)
			elif postprocess == 'add':
				return_ = return_fw + return_bw
			elif postprocess == 'max':
				return_ = tf.maximum(return_fw, return_bw)
			elif postprocess == 'linear':
				if out_dim is None:
					out_dim = 2 * hidden_size
					return_ = tf.concat([return_fw, return_bw], 2 if return_outputs else 1)
					return_ = tf.layers.dense(return_, out_dim)
			else:
				return_ = postprocess(return_fw, return_bw)
			return_list.append(return_)
		if len(return_list) == 1:
			return return_list[0]
		return return_list
	
def attention_rnn( inputs_list, sequence_length_list,
			 mask_is_length=True,
			 logit_fn=None,
			 scale_dot=False,
			 normalizer=tf.nn.softmax,
			 tensors=None,
			 scope=None,
			 reuse=False):
	"""
	attention encoder to acquire sell-attented setence embedding
	"""
	if not isinstance(inputs_list, list):
		inputs_list = [inputs_list]
	if sequence_length_list is None:
		sequence_length_list = [None] * len(inputs_list)
	elif not isinstance(sequence_length_list, list):
		sequence_length_list = [sequence_length_list]
	
	assert len(inputs_list[0].get_shape()) == 3, 'The rank of `tensor` must be 3 but got {}.'.format(len(inputs_list[0].get_shape()))
	# inputs_list[0] is the batch_size * seq_len * word_dim
	
	max_sentence_len = inputs_list[0].get_shape()[1] # seq_len
	print("------------max sequence length------------", max_sentence_len)
	hidden_dim = inputs_list[0].get_shape()[-1] # hidden_dim_dim
	return_list = []
	with tf.variable_scope(scope or 'att_encoder', reuse=reuse) as vs:	
		for inputs, sequence_lengths in zip(inputs_list, sequence_length_list):
			if return_list:
				vs.reuse_variables()
			if mask_is_length == False:
				inputs_mask = tf.sequence_mask(sequence_lengths, # sequence_length [batch_size]
								maxlen=max_sentence_len,
								dtype=tf.bool,
								name="bool_mask")
			else:
				inputs_mask = sequence_lengths
			
			inputs_attention = self_att(tensor=inputs, 
							   mask=inputs_mask, 
							   mask_is_length=mask_is_length, 
							   logit_fn=logit_fn, 
							   scale_dot=scale_dot, 
							   normalizer=normalizer, 
							   scope=scope, 
							   reuse=reuse)
		
			# inputs = [batch_size, seq_len, hidden_dim]
			# input_attention = [batch_size, hidden_dim]
			return_list.append(inputs_attention)
		if len(return_list) == 1:
			return return_list[0]
		return return_list
	
def hierarchical_encoder(hidden_size, inputs_list, turn_length_vector,
				history_state=None,attention=False,
				mask_is_length=True,
				logit_fn=None,
				scale_dot=False,
				normalizer=tf.nn.softmax,
				scope=None,
				reuse=False):
	"""
	input_list is a list of tensor [batch_size, 1, hidden_dim]
	len(input_list) is the maximize turn or sentence in a document
	"""
	if not isinstance(inputs_list, list):
		inputs_list = [inputs_list]
	
	max_turn_len = len(inputs_list)
	with tf.variable_scope(scope or 'hierarchical_encoder', reuse=reuse) as vs:
		history_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
		
		batch_size = inputs_list[0].get_shape()[0]
		
		input_tensor = tf.stack(inputs_list, axis=1, name='stack') # [batch_size, max_sentence_len, hidden_dim]
		
		print(input_tensor.shape)
		#initial_state = history_cell.zero_state(batch_size, dtype=tf.float32)
		initial_state = tf.contrib.rnn.LSTMStateTuple(history_state, history_state)
		print(type(initial_state),"------------------")
		o_history, state_history = tf.nn.dynamic_rnn(history_cell,
								input_tensor,
								sequence_length=turn_length_vector,
								initial_state=initial_state,
								dtype='float',
								scope='rnn_history')
		
		if attention:
			
			if mask_is_length == False:
				inputs_mask = tf.sequence_mask(turn_length_vector, # sequence_length [batch_size]
								maxlen=max_turn_len,
								dtype=tf.bool,
								name="bool_mask")
			else:
				inputs_mask = turn_length_vector
			
			attened_o_history = self_att(tensor=o_history, 
							   mask=inputs_mask, 
							   mask_is_length=mask_is_length, 
							   logit_fn=logit_fn, 
							   scale_dot=scale_dot, 
							   normalizer=normalizer,
							   scope=scope, 
							   reuse=reuse)
			
			return (o_history, attened_o_history, state_history)
		else:
			last_hidden_state = last_relevant_output(o_history, turn_length_vector)
			return (o_history, last_hidden_state, state_history)
		
def inference_history(history_cell, inputs_list, sequence_length_list,
				history_state=None,
				attention=False,
				mask_is_length=True,
				logit_fn=None,
				scale_dot=False,
				normalizer=tf.nn.softmax,
				scope=None,
				reuse=False):
	
	if not isinstance(inputs_list, list):
		inputs_list = [inputs_list]
	#if sequence_length_list is None:
	#	sequence_length_list = [None] * len(inputs_list)
	#elif not isinstance(sequence_length_list, list):
	#	sequence_length_list = [sequence_length_list]
	#print("-------type of sequence_length_list---", type(sequence_length_list))
	if not attention:
		with tf.variable_scope(scope or 'inference_history_encoder', reuse=reuse) as vs:
			"""
			if history_cell.state_size is tuple, initial_state is a tuple for hidden state and memory state
			if history_cell.state_size is int, initial_state is a tensor for hidden state and memroty state
			"""
			if history_cell._state_is_tuple:
				print("---------------tuple of hidden state--------")
				history_output, history_encoder_state = tf.contrib.rnn.static_rnn(history_cell, [inputs_list[-1]], 
													initial_state=(history_state,history_state),
													dtype=tf.float32)
			else:
				history_output, history_encoder_state = tf.contrib.rnn.static_rnn(history_cell, [inputs_list[-1]], 
													initial_state=history_state,
													dtype=tf.float32)
							
			return (history_output[0], history_encoder_state)
	else:
		# from begin to current time step
		with tf.variable_scope(scope or 'inference_history_encoder', reuse=reuse) as vs:
			input_tensor = tf.stack(inputs_list, axis=1, name='stack_input') # [batch_size x time_step x data_dim]
			
			batch_size = input_tensor.get_shape()[0]
			
			length_tensor = tf.squeeze(sequence_length_list)
			initial_state = history_state#history_cell.zero_state(batch_size, dtype=tf.float32)
			
			o_history, state_history = tf.nn.dynamic_rnn(history_cell,
								input_tensor,
								sequence_length=length_tensor,
								initial_state=(initial_state,initial_state),
								dtype='float',
								scope='rnn_history')
			
			attened_o_history = self_att(tensor=o_history, 
							   mask=None, 
							   mask_is_length=mask_is_length, 
							   logit_fn=logit_fn, 
							   scale_dot=scale_dot, 
							   normalizer=normalizer,
							   scope=scope, 
							   reuse=reuse)
			return (attened_o_history, state_history)
				   
def self_att(tensor,
			 mask=None,
			 mask_is_length=True,
			 logit_fn=None,
			 scale_dot=False,
			 normalizer=tf.nn.softmax,
			 scope=None,
			 reuse=False):
	"""Performs self attention.

		Performs self attention to obtain single vector representation for a sequence of vectors.

	Args:
		tensor: [batch_size, sequence_length, hidden_size]-shaped tensor
		mask: Length mask (shape of [batch_size]) or boolean mask ([batch_size, sequence_length])
		mask_is_length: `True` if `mask` is length mask, `False` if it is boolean mask
		logit_fn: `logit_fn(tensor)` to obtain logits.
		scale_dot: `bool`, whether to scale the dot product by dividing by sqrt(hidden_size).
		normalizer: function to normalize logits.
		scope: `string` for defining variable scope
		reuse: Reuse if `True`.
	Returns:
		[batch_size, hidden_size]-shaped tensor.
	"""
	assert len(tensor.get_shape()) == 3, 'The rank of `tensor` must be 3 but got {}.'.format(len(tensor.get_shape()))
	with tf.variable_scope(scope or 'self_att', reuse=reuse):
		hidden_size = tensor.get_shape().as_list()[-1]
		if logit_fn is None:
			logits = tf.layers.dense(tensor, hidden_size, activation=tf.tanh)
			logits = tf.squeeze(tf.layers.dense(logits, 1), 2)
		else:
			logits = logit_fn(tensor)
		if scale_dot:
			logits /= tf.sqrt(hidden_size)
		if mask is not None:
			logits = exp_mask(logits, mask, mask_is_length=mask_is_length)
		weights = normalizer(logits)
		out = tf.reduce_sum(tf.expand_dims(weights, -1) * tensor, 1)
		return out

def highway(inputs,
			outputs=None,
			dropout_rate=0.0,
			batch_norm=False,
			training=False,
			scope=None,
			reuse=False):
	"""Single-layer highway networks (https://arxiv.org/abs/1505.00387).

	Args:
		inputs: Arbitrary-rank `float` tensor, where the first dim is batch size
		  and the last dim is where the highway network is applied.
		outputs: If provided, will replace the perceptron layer (i.e. gating only.)
		dropout_rate: `float` value, input dropout rate.
		batch_norm: `bool` value, whether to use batch normalization.
		training: `bool` value, whether the current run is training.
		scope: `str` value variable scope, default to `highway_net`.
		reuse: `bool` value, whether to reuse variables.
	Returns:
		The output of the highway network, same shape as `inputs`.
	"""
	with tf.variable_scope(scope or 'highway', reuse=reuse):
		inputs = tf.nn.dropout(inputs, dropout_rate)
		dim = inputs.get_shape()[-1]
		if outputs is None:
			outputs = tf.layers.dense(inputs, dim, name='outputs')
		if batch_norm:
			outputs = tf.layers.batch_normalization(outputs, training=training)
		outputs = tf.nn.relu(outputs)
		gate = tf.layers.dense(inputs, dim, activation=tf.nn.sigmoid, name='gate')
		return gate * inputs + (1 - gate) * outputs

def highway_net(inputs,
				num_layers,
				dropout_rate=0.0,
				batch_norm=False,
				training=False,
				scope=None,
				reuse=False):
	"""Multi-layer highway networks (https://arxiv.org/abs/1505.00387).

	Args:
		inputs: `float` input tensor to the highway networks.
		num_layers: `int` value, indicating the number of highway layers to build.
		dropout_rate: `float` value for the input dropout rate.
		batch_norm: `bool` value, indicating whether to use batch normalization
		  or not.
		training: `bool` value, indicating whether the current run is training
		 or not (e.g. eval or inference).
		scope: `str` value, variable scope. Default is `highway_net`.
		reuse: `bool` value, indicating whether the variables in this function
		  are reused.
	  Returns:
		The output of the highway networks, which is the same shape as `inputs`.
	"""
	with tf.variable_scope(scope or 'highway_net', reuse=reuse):
		outputs = inputs
		for i in range(num_layers):
			outputs = highway(
			  outputs,
			  dropout_rate=dropout_rate,
			  batch_norm=batch_norm,
			  training=training,
			  scope='layer_{}'.format(i))
		return outputs