import tensorflow as tf
import math
from functools import reduce
from operator import mul
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, RNNCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def normalize_layer(inputs, epsilon=1e-8, scope=None):
    with tf.variable_scope(scope or "layer_name"):
        input_shape = inputs.get_shape()
        param_shape = input_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(param_shape))
        gamma = tf.Variable(tf.ones(param_shape))
        normalized = (inputs - mean)/((variance + epsilon) ** 0.5)
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
        return outputs


def highway_layer(inputs, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        hidden_units = inputs.get_shape().as_list()[-1]
        with tf.variable_scope("trans"):
            trans = tf.layers.dropout(inputs, rate=1.0-keep_prob, training=is_train)
            trans = tf.layers.dense(trans, units=hidden_units, use_bias=use_bias,
                                    bias_initializer=tf.constant_initializer(bias_init), activation=None)
            trans = tf.nn.relu(trans)
        with tf.variable_scope("gate"):
            gate = tf.layers.dropout(inputs, rate=1.0-keep_prob, training=is_train)
            gate = tf.layers.dense(gate, units=hidden_units, use_bias=use_bias,
                                   bias_initializer=tf.constant_initializer(bias_init), activation=None)
            gate = tf.nn.sigmoid(gate)
        outputs = gate * trans + (1 - gate) * inputs
        return outputs


def highway_network(inputs, highway_layers=2, use_bias=True, bias_init=0.0, keep_prob=1.0, is_train=False, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        previous = inputs
        current = None
        for layer in range(highway_layers):
            current = highway_layer(previous,
                                    use_bias,
                                    bias_init,
                                    keep_prob,
                                    is_train,
                                    scope="highway_layer_".format(layer))
            previous = current
        return current


def conv1d(inputs, filter_size, channel_size, padding, is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = inputs.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, channel_size, num_channels, filter_size], dtype=tf.float32)
        bias_ = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        inputs = tf.layers.dropout(inputs, rate=drop_rate, training=is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        x = tf.nn.conv2d(inputs, filter_, strides, padding) + bias_
        outputs = tf.reduce_max(tf.nn.relu(x, axis=2))
        return outputs


def multi_conv1d(inputs, filter_sizes, channel_sizes, padding="VALID", is_train=True, drop_rate=0.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(channel_sizes)
        outputs = []
        for i, (filter_size, channel_size) in enumerate(zip(filter_sizes, channel_sizes)):
            if filter_size == 0:
                continue
            output = conv1d(inputs,
                            filter_size,
                            channel_size,
                            padding,
                            is_train,
                            drop_rate,
                            "conv1d_{}".format(i))
            outputs.append(output)
        concatenated = tf.concat(axis=2, values=outputs)
        return concatenated


class BiRNN:
    def __init__(self, num_units, cell_type='lstm', scope=None):
        self.cell_fw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.cell_bw = GRUCell(num_units) if cell_type == 'gru' else LSTMCell(num_units)
        self.scope = scope or "bi_rnn"

    def __call__(self, inputs, seq_len, use_last_state=False, time_major=False):
        assert not time_major, "BiRNN can't support time_major"
        with tf.variable_scope(self.scope):
            flat_inp = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [seq_len]
            outputs, ((_, output_state_fw), (_, output_state_bw)) = bidirectional_dynamic_rnn(self.cell_fw,
                                                                                              self.cell_bw, flat_inp,
                                                                                              sequence_length=seq_len,
                                                                                              dtype=tf.float32)
            if use_last_state:
                output = tf.concat([output_state_fw, output_state_bw], axis=-1)     # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)    # remove max_time shape
            else:
                output = tf.concat(outputs, axis=-1)     # shape = [-1, max_time, num_units]
            return output


class DenselyConnectedBiRNN:
    def __init__(self, num_layers, num_units, cell_type='lstm', scope=None):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its length should be same as num_layers"
        self.dense_bi_rnn = []
        for i in range(num_layers):
            units = num_units[i] if type(num_units) is list else num_units
            self.dense_bi_rnn.append(BiRNN(units, cell_type, scope="bi_rnn_{}".format(i)))
        self.num_layers = num_layers
        self.scope = scope or "densely_connected_bi_rnn"

    def __call__(self, inputs, seq_len, time_major=False):
        """
        :param inputs:The RNN inputs. If time_major == False (default), this must
        be a tensor of shape: [batch_size, max_time, ...],
        or a nested tuple of such elements. If time_major == True, this must be a
        tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.

        :param seq_len:An int32/int64 vector, size [batch_size], containing the
            actual lengths for each of the sequences in the batch.
        """
        assert not time_major, "DenseConnectBiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flatten_inputs = flatten(inputs, keep=2)      # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)            # reshape [seq_len]
            curr_in = flatten_inputs
            for i in range(self.num_layers):
                curr_out = self.dense_bi_rnn[i](curr_in, seq_len)
                if i < self.num_layers - 1:
                    curr_in = tf.concat([curr_in, curr_out], axis=-1)
                else:
                    curr_in = curr_out
            outputs = reconstruct(curr_in, ref=inputs, keep=2)
            return outputs


def multi_head_attention(queries, keys, num_heads, attention_size, drop_rate=0.0, is_train=True, reuse=None,
                         scope=None):
    with tf.variable_scope(scope or "multi_head_attention"):
        if attention_size is None:
            attention_size = queries.get_shape().as_list()

        # linear projections, shape=(batch_size, max_time, attention_size)
        query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_projection")
        key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_projection")
        value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_projection")

        # split and concatenate, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)

        # matmul (Q K(T))
        att_out = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))

        # scale (QK(T)/ sqrt(d(k))
        att_out = att_out / (query_.get_shape().as_list()[-1] ** 0.5)

        # masking for key
        # shape=(batch_size, max_time)
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        # shape = (batch_size * num_heads, max_time)
        key_masks = tf.tile(key_masks, [num_heads, 1])
        # shape = (batch_size * num_heads, max_time, max_time)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(att_out) * (-2 ** 32 + 1)
        # shape=(batch_size, max_time, attention_size)
        att_out = tf.where(tf.equals(key_masks, 0), paddings, att_out)

        att_out = tf.nn.softmax(att_out)

        # query_masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, 1), [1, tf.shape(keys)[1], 1])
        att_out *= query_masks
        att_out = tf.layers.dropout(att_out, rate=drop_rate, training=is_train)

        # weighted sum
        outputs = tf.matmul(att_out, value_)

        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # residual connection
        outputs += queries
        outputs = normalize_layer()
        return outputs


def dot_attention(inputs, memory, hidden, drop_rate=0.0, is_train=True, scope=None):
    drop_inputs = tf.layers.dropout(inputs, rate=drop_rate, training=is_train)
    drop_memory = tf.layers.dropout(memory, rate=drop_rate, training=is_train)

    with tf.variable_scope("attention"):
        input_ = tf.layers.dense(drop_inputs, hidden, use_bias=False, name='memory')
        input_ = tf.nn.relu(input_)
        memory_ = tf.layers.dense(drop_memory, hidden, use_bias=True, name='memory')
        memory_ = tf.nn.relu(memory_)
        outputs = tf.matmul(input_, tf.transpose(memory_[0, 2, 1])/ (hidden ** 0.5))
        logits = tf.nn.softmax(outputs)
        outputs = tf.matmul(logits, outputs)
        res = tf.concat([inputs, outputs], axis=-1)

    with tf.variable_scope("gate"):
        dim = res.get_shape().as_list()[-1]
        d_res = tf.layers.dropout(res, rate=drop_rate, training=is_train)
        d_res = tf.layers.dense(d_res, dim, use_bias=False, name='gate')
        d_res = tf.nn.relu(d_res)
        gate = tf.nn.sigmoid(d_res)
        return res * gate


class AttentionCell(RNNCell):
    """
    refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, num_units, memory, pmemory, cell_type='lstm'):
        self.num_units = num_units
        self._cell = GRUCell(self.num_units) if cell_type == 'gru' else LSTMCell(self.num_units)
        self.memory = memory
        self.pmemory = pmemory
        self.memory_units = memory.get_shape().as_list()[-1]

    @property
    def state_size(self):
        self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        c, memory = state
        h_attn = tf.layers.dense(memory, self.memory_units, use_bias=False, name='wah')
        # (max_time, batch_size, attention_unit
        h_a = tf.nn.tanh(tf.add(self.pmemory, h_attn))
        alphas = tf.squeeze(tf.exp(tf.layers.dense(h_a, units=1, use_bias=False, name='way')), axis=[-1])
        # (max_time, batch_size)
        alphas = tf.div(alphas, tf.reduce_sum(alphas, axis=0, keep_dims=True))
        # (batch_time, attn_units)
        w_context = tf.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        lfc = tf.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')
        # (batch_size, num_units)
        fused = tf.nn.sigmoid(tf.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                              tf.layers.dense(h, self.num_units, name='wfh'))
        h_fused = tf.multiply(fused, lfc) + h
        return h_fused, new_state


def label_smoothing(inputs, epsilon=0.1):
    dim = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / dim)


def flatten(tensor, keep):
    tensor_shape = tensor.get_shape().as_list()
    start = len(tensor_shape) - keep
    left = reduce(mul, [tensor_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    output_shape = [left] + [tensor_shape or tf.shape(tensor)[i] for i in range(start, len(tensor_shape))]
    flattened = tf.reshape(tensor, output_shape)
    return flattened


def reconstruct(tensor, ref, keep, remove_shape=None):
    tensor_shape = tensor.get_shape().as_list()
    ref_shape = ref.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    if remove_shape is not None:
        tensor_start = tensor_start + remove_shape
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    out_shape = pre_shape + keep_shape
    output = tf.reshape(tensor, shape=out_shape)
    return output


