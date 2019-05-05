import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import viterbi_decode, crf_log_likelihood
from tensorflow.python.ops.rnn_cell import LSTMCell, GRUCell, MultiRNNCell
from utilities import CoNLLeval, load_dataset, get_logger, pre_process_batch, align_data
from utilities.common_utils import word_conversion, UNK
import os


class BaseModel:
    def __init__(self, config):
        self.conf = config
        self._initialise_config()
        self.saver, self.sess = None, None
        self._add_placeholders()
        self._build_embedding_op()
        self._build_model_op()
        self._build_loss_op()
        self._build_train_op()
        print('params number {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        self.initialise_session()

    def _initialise_config(self):
        if not os.path.exists(self.conf["checkpoint_path"]):
            os.makedirs(self.conf["checkpoint_path"])
        if not os.path.exists(self.conf["summary_path"]):
            os.makedirs(self.conf["summary_path"])
        self.logger = get_logger(os.path.join(self.conf["checkpoint_path"] + "log.txt"))
        # load dictionary
        dict_ = load_dataset(self.conf["vocab"])
        self.word_dict, self.char_dict, self.tag_dict = dict_["word_dict"], dict_["char_dict"], dict_["tag_dict"]
        del dict_
        self.word_vocab_size = len(self.word_dict)
        self.char_vocab_size = len(self.char_dict)
        self.tag_vocab_size = len(self.tag_dict)
        self.rev_word_dict = dict([(idx, word) for (word, idx) in self.word_dict.items()])
        self.rev_char_dict = dict([(idx, char) for (char, idx) in self.char_dict.items()])
        self.rev_tag_dict = dict([(idx, tag) for (tag, idx) in self.tag_dict.items()])

    def initialise_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=self.conf["max_to_keep"])
        self.sess.run(tf.global_variables_initializer())

    def restore_last_sess(self, ckpt_path=None):
        if ckpt_path is not None:
            # Returns CheckpointState proto from the "checkpoint" file.
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.conf["checkpoint_path"])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_sess(self, epoch):
        self.saver.save(self.sess, self.conf["checkpoint_path"] + self.conf["model_name"], global_step=epoch)

    def close_sess(self):
        self.sess.close()

    def _add_summary(self):
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.conf["summary_path"] + "train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.conf["summary_path"] + "test")

    def reinitialise_weights(self, scope_name=None):
        if scope_name is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            variables = tf.contrib.framework.get_variables(scope_name)
            self.sess.run(tf.variables_initializer(variables))

    @staticmethod
    def variable_summaries(variable, name=None):
        with tf.name_scope(name or "summaries"):
            mean = tf.reduce_mean(variable)
            # Outputs a Summary protocol buffer containing a single scalar value.
            tf.summary.scalar("mean", mean)     # add mean to summary
            std_dev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))  # calculate standard_deviation
            tf.summary.scalar("std_dev", std_dev)
            tf.summary.scalar("min", tf.reduce_min(variable))
            tf.summary.scalar("max", tf.reduce_max(variable))
            tf.summary.histogram("histogram", variable)

    @staticmethod
    def viterbi_decoding(logits, trans_param, seq_len):
        viterbi_sequences = []
        for logit, length in zip(logits, seq_len):
            logit = logit[:length]
            viterbi, viterbi_score = viterbi_decode(logit, trans_param)
            viterbi_sequences += [viterbi]
        return viterbi_sequences

    def _create_single_rnn_cell(self, num_units):
        cell = GRUCell(num_units) if self.conf["cell_type"] == "gru" else LSTMCell(num_units)
        return cell

    def _create_rnn_cell(self):
        if self.conf["num_layers"] is None or self.conf["num_layers"] <= 1:
            return self._create_single_rnn_cell(self.conf["num_units"])
        else:
            MultiRNNCell([self._create_single_rnn_cell(self.conf["num_units"]) for _ in self.conf["num_layers"]])

    def _add_placeholders(self):
        raise NotImplementedError("To be implemented...")

    def _get_feed_dict(self, data):
        raise NotImplementedError("To be implemented...")

    def _build_embedding_op(self):
        raise NotImplementedError("To be implemented...")

    def _build_model_op(self):
        raise NotImplementedError("To be implemented...")

    def _build_loss_op(self):
        if self.conf["use_crf"]:
            crf_likelihood, self.trans_params = crf_log_likelihood(self.logits, self.tags, self.seq_len)
            self.loss = tf.reduce_mean(crf_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
            mask = tf.sequence_mask(self.seq_len)
            self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
        tf.summary.scalar("loss", self.loss)

    def _build_train_op(self):
        with tf.variable_scope("train_step"):
            if self.conf["optimizer"] == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.conf["optimizer"] == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.conf["optimizer"] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.conf["optimizer"] == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            else:
                if self.conf["optimizer"] != 'adam':
                    print("Unsupported optimizer {}. Using Adam".format(self.conf["optimizer"]))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.conf["grad_clip"] is not None and self.conf["grad_clip"] > 0:
            grads, variables = zip(*optimizer.compute_gradients(self.loss))
            grads, _ = tf.clip_by_global_norm(grads)
            self.train_op = optimizer.apply_gradients(zip(grads, variables))
        else:
            self.train_op = optimizer.minimize(self.loss)

    def train_epoch(self, train_set, valid_data, epoch):
        raise NotImplementedError("To be implemented in child class..")

    def _predict_op(self, data):
        feed_dict = self._get_feed_dict(data)
        if self.conf["use_crf"]:
            logits, trans_params, seq_len = self.sess.run([self.logits, self.trans_params, self.seq_len],
                                                          feed_dict=feed_dict)
            return self.viterbi_decoding(logits, trans_params, seq_len)
        else:
            pred_logits = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            logits = self.sess.run(pred_logits, feed_dict=feed_dict)
            return logits

    def train(self, train_set, valid_data, valid_set, test_set):
        self.logger.info("Start Training...")
        best_f1_score, no_improve_epoch, init_lr = -np.inf, 0, self.conf["lr"]
        for epoch in range(1, self.conf["epochs"] + 1):
            self.logger.info("Epoch {}/{}".format(epoch, self.conf["epochs"]))
            self.train_epoch(train_set, valid_data, epoch)
            self.evaluate(valid_set, "dev")
            score = self.evaluate(test_set, "test")
            if self.conf["use_lr_decay"]:
                self.conf["lr"] = max(init_lr / (1 + self.conf["lr_decay"] * epoch), self.conf["minimal_lr"])
            if score["FB1"] > best_f1_score:
                best_f1_score = score["FB1"]
                no_improve_epoch = 0
                self.save_sess(epoch)
                self.logger.info("--New best score on test set: {:04.2f}".format(best_f1_score))
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= self.conf["no_imprv_tolerance"]:
                    self.logger.info("Early Stopping on {}th epoch due to no improvement,best score: {:04.2f}"
                                     .format(epoch, best_f1_score))
                    break
        self.train_writer.close()
        self.test_writer.close()

    def evaluate(self, dataset, name):
        save_path = os.path.join(self.conf["checkpoint_path"], "result.txt")
        predictions, true_tags, word_list = list(), list(), list()
        for data in dataset:
            predict = self._predict_op(data)
            for tags, preds, words, seq_len in zip(data["tags"], predict, data["words"], data["seq_len"]):
                tags = [self.rev_tag_dict[idx] for idx in tags[:seq_len]]
                words = [self.rev_word_dict[idx] for idx in words[:seq_len]]
                preds = [self.rev_tag_dict[idx] for idx in preds[:seq_len]]
                predictions.append(preds)
                word_list.append(words)
                true_tags.append(tags)
        c_eval = CoNLLeval()
        score = c_eval.conlleval(predictions, true_tags, word_list, save_path)
        self.logger.info("{} dataset -- acc: {:04.2f}, pre: {:04.2f}, rec: {:04.2f}, FB1: {:04.2f}"
                         .format(name, score["accuracy"], score["precision"], score["recall"], score["FB1"]))
        return score

    def word_to_indices(self, words):
        """
            Convert input words into batchnized word/chars indices for inference
            :param words: input words
            :return: batchnized word indices
        """
        char_idx = []
        for word in words:
            idxs = [self.char_dict[char] if char in self.char_dict else self.char_dict[UNK] for char in word]
            char_idx.append(idxs)
        words = [word_conversion(word) for word in words]
        word_idx = [self.word_dict[word] if word in self.word_dict else self.word_dict[UNK] for word in words]
        return pre_process_batch([word_idx], [char_idx])

    def inference(self, sentence):
        sentence = sentence.lstrip().rstrip().split(" ")
        data = self.word_to_indices(sentence)
        predicts = self._predict_op(data)
        predicts = [self.rev_word_dict[idx] for idx in list(predicts[0])]
        results = align_data({"inputs": sentence, "outputs": predicts})
        print("{}\n{}".format(results["inputs"], results["outputs"]))
