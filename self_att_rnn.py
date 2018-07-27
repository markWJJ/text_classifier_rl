import rnn_utils
import tensorflow as tf
from model_template import ModelTemplate
import encoder

class SelfAttRNN(ModelTemplate):
	def __init__(self):
		super(SelfAttRNN, self).__init__()

	def build_model(self):
		self.build_op()

	def build_network(self):
		self.options = self.config["options"]
		self.hidden_units = self.options["context_lstm_dims"]
		self.highway_layer_nums = self.options["highway_layer_num"]
		self.wd = self.config.get("weight_decay", None)

		with tf.variable_scope(self.scope+'intent_embedding'):

			input_rnn = encoder.bi_rnn(self.hidden_units,
									self.s1_emb,
									sequence_length_list=self.sent1_token_len,
									scope="rnn_encoding",
									dropout_rate=self.dropout_keep_prob,
									training=self.is_train,
									stack=False,
									cells=None,
									postprocess='concat',
									return_outputs=True,
									out_dim=None,
									reuse=False)

			atten_encoding = encoder.attention_rnn(input_rnn, self.sent1_token_len,
									mask_is_length=True,
									logit_fn=None,
									scale_dot=False,
									normalizer=tf.nn.softmax,
									tensors=None,
									scope="self_att",
									reuse=False)

			self.output_features = encoder.highway_net(atten_encoding,
								self.highway_layer_nums,
								dropout_rate=self.dropout_keep_prob,
								batch_norm=False,
								training=self.is_train,
								scope="highway",
								reuse=False)

			self.estimation = tf.layers.dense(inputs=self.output_features, units=self.num_classes)

			self.pred_probs = tf.contrib.layers.softmax(self.estimation)
			self.logits = tf.cast(tf.argmax(self.estimation, 1, name="predictions"), tf.int32)
			rnn_utils.add_reg_without_bias()


	def build_loss(self):

		with tf.name_scope("loss"):
			wd_loss = 0
			print("------------wd--------------", self.wd)
			if self.wd is not None:
				for var in set(tf.get_collection('reg_vars', self.scope)):
					weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
										  name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
					wd_loss += weight_decay
				print("---------using l2 regualarization------------")

			if self.config["loss_type"] == "cross_entropy":
				self.batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.gold_label, name="batch_loss")
				self.loss = tf.add(tf.reduce_mean(self.batch_loss), wd_loss)
                
	   
		tf.add_to_collection('ema/scalar', self.loss)
		print("List of Variables:")
		for v in tf.trainable_variables():
			print(v.name)

	def build_accuracy(self):
		correct = tf.equal(
			self.logits,
			self.gold_label
		)
		self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	def get_feed_dict(self, sample_batch, dropout_keep_prob, data_type='train'):
		if data_type == "train" or data_type == "test":
			[sent1_token_b, gold_label_b, sent1_token_len] = sample_batch

			feed_dict = {
                self.sent1_token: sent1_token_b,
                self.gold_label:gold_label_b,
                self.is_train: True if data_type == 'train' else False,
                self.learning_rate: self.learning_rate_value,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0
			}
		elif data_type == "infer":
			[sent1_token_b, _, sent1_token_len] = sample_batch

			feed_dict = {
                self.sent1_token: sent1_token_b,
                self.is_train: False,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0
			}
		return feed_dict  

			







