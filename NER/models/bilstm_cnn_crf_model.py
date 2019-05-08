import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from models import BaseModel, multi_head_attention, multi_conv1d
from models import normalize_layer, highway_network
from utilities import Progbar


class SeqLabelModel(BaseModel):
    def __init__(self, conf):
        super(SeqLabelModel, self).__init__(conf)

    def _add_placeholders(self):
        self.words = tf.placeholder(dtype=tf.int32, shape=[None, None], name="words")
        self.tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tags")
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None, None], name="seq_len")
        if self.conf["use_chars"]:
            self.chars = tf.placeholder(dtype=tf.int32, shape=[None, None], name="chars")
            self.char_seq_len = tf.placeholder(dtype=tf.int32, shape=[None, None], name="char_seq_len")
        self.is_train = tf.placeholder(tf.bool, shape=[], name="is_train")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.drop_rate = tf.placeholder(tf.float32, name="drop_rate")
        self.lr = tf.placeholder(tf.float32, name="learning")

    def _get_feed_dict(self, batch, keep_prob=1.0, is_train=False, lr=None):
        feed_dict = {self.words: batch["words"], self.seq_len: batch["seq_len"], self.batch_size: batch["batch_size"]}
        if "tags" in batch:
            feed_dict[self.tags] = batch["tags"]
        if self.conf["use_chars"]:
            feed_dict[self.chars] = batch["chars"]
            feed_dict[self.char_seq_len] = batch["char_seq_len"]
        feed_dict[self.is_train] = is_train
        feed_dict[self.keep_prob] = keep_prob
        feed_dict[self.drop_rate] = 1.0 - keep_prob
        if lr is not None:
            feed_dict[self.lr] = lr
        return feed_dict

    def _create_rnn_cell(self):
        if self.conf["num_layers"] is None or self.conf["nym_layers"] <= 1:
            return self._create_single_rnn_cell(self.conf["num_units"])
        else:
            if self.conf["use_stack_rnn"]:
                return [self._create_single_rnn_cell(self.conf["num_units"]) for _ in range(self.conf["num_layers"])]
            else:
                MultiRNNCell([self._create_single_rnn_cell(self.conf["num_units"]
                                                           for _ in range(self.conf["num_layers"]))])

    def _build_embedding_op(self):
        with tf.variable_scope("embedding"):
            if self.conf["use_pretrained"]:
                self.word_embeddings = tf.Variable(np.load(self.conf["pretrained_emb"])["embedding"],
                                                   name="emb", dtype=tf.int32, trainable=self.conf["tuning_emb"])
            else:
                self.word_embeddings = tf.get_variable(name="emb", trainable=True, dtype=tf.float32,
                                                       shape=[self.word_vocab_size, self.conf["emb_dim"]])
            word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.words, name="word_emb")
            print("word embedding shape = {}".format(word_emb.get_shape().as_list()))
            if self.conf["use_chars"]:
                self.char_embeddings = tf.get_variable(name="char_emb", trainable=True, dtype=tf.float32,
                                                       shape=(self.char_vocab_size, self.conf["char_emb_dim"]))
                char_emb = tf.nn.embedding_lookup(self.char_embeddings)
                char_rep = multi_conv1d(char_emb, self.conf["filter_sizes"], self.conf["channel_sizes"],
                                        drop_rate=self.drop_rate, is_train=self.is_train)
                print("char representation shape = {}".format(char_rep.get_shape().as_list()))
                word_emb = tf.concat([word_emb, char_rep], axis=-1)
            if self.conf["use_highway"]:
                self.word_emb = highway_network(word_emb, self.conf["highway_layers"], use_bias=True,
                                                is_train=self.is_train, bias_init=0.0, keep_prob=self.keep_prob)
            else:
                self.word_emb = tf.layers.dropout(word_emb, rate=self.drop_rate, training=self.is_train)
            print("words and chars concat representation shape {}".format(self.word_emb.get_shape().as_list()))

    def _build_model_op(self):
        pass

    def train_epoch(self, train_set, valid_data, epoch):
        pass
