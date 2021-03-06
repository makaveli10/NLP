# -*- coding: utf-8 -*-
"""tf_nn_lm_batch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15h3Bhb6WxKDBAuHcy7bXNXDltth-b77D
"""

import tensorflow as tf
import numpy as np
import math
import time
import random
from collections import defaultdict
from tensorflow.keras import layers

# The length of the n-gram
N = 2 # The length of the n-gram
EMB_SIZE = 128 # The size of the embedding
HID_SIZE = 128 # The size of the hidden layer


# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already converted
#       into an easy-to-use format with "<unk>" symbols. If we were using other
#       data we would have to do pre-processing and consider how to choose
#       unknown words, etc.
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            yield [w2i[x] for x in line.strip().split(" ")]

# Read in the data
train = list(read_dataset("train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)

class FFN_LM(tf.keras.Model):
    def __init__(self, nwords, emb_size, hid_size, num_hist, dropout):
        super(FFN_LM, self).__init__()
        self.embedding = layers.Embedding(nwords, emb_size)
        self.fnn = tf.keras.Sequential()
        self.fnn.add(layers.Dense(hid_size, activation='tanh'))
        self.fnn.add(layers.Dropout(dropout))
        self.fnn.add(layers.Dense(nwords))

    def call(self, inputs):
        # 3D Tensor of size [batch_size x num_hist x emb_size]
        emb_out = self.embedding(inputs) 
        # 2D Tensor of size [batch_size x (num_hist * emb_size)]
        emb_view =  tf.reshape(emb_out, [tf.shape(emb_out)[0], -1])
        # 2D Tensor of size [batch_size x nwords]
        out = self.fnn(emb_view)
        return out

# Initialize the model and the optimizer
model = FFN_LM(nwords=nwords,
               emb_size=EMB_SIZE,
               hid_size=HID_SIZE,
               num_hist=N,
               dropout=0.2)
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def convert_to_variables(words):
    var = tf.constant(words)
    return var

# A function to calculate scores for one value
def calc_score_of_histories(words):
    # This will change from a list of histories, to a pytorch Variable whose data type is LongTensor
    words_var = convert_to_variables(words)
    logits = model(words_var)
    return logits

@tf.function(experimental_relax_shapes=True)
def step(words_var, all_targets):
    with tf.GradientTape() as tape:
        logits = model(words_var)
        loss = loss_fn(all_targets, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function(experimental_relax_shapes=True)
def valid_step(words_var, all_targets):
    logits = model(words_var)
    loss = loss_fn(all_targets, logits)
    return loss

# Calculate the loss value for the entire sentence

def get_words_and_targs(sent):
       # The initial history is equal to end of sentence symbols
    hist = [S] * N
    # Step through the sentence, including the end of sentence token
    all_histories = []
    all_targets = []
    for next_word in sent + [S]:
        all_histories.append(list(hist))
        all_targets.append(next_word)
        hist = hist[1:] + [next_word]
    words_var = convert_to_variables(all_histories)
    all_targets = convert_to_variables(all_targets)
    return words_var, all_targets


def calc_sent_loss(sent):
    words_var, all_targets = get_words_and_targs(sent) 
    loss = step(words_var, all_targets)
    return loss

# Calculate the loss value for the entire sentence
def calc_valid_sent_loss(sent):
    words_var, all_targets = get_words_and_targs(sent) 
    loss = valid_step(words_var, all_targets)
    return loss

MAX_LEN = 100
# Generate a sentence
def generate_sent():
    hist = [S] * N
    sent = []
    while True:
        logits = calc_score_of_histories([hist])
        prob = tf.keras.activations.softmax(logits)
        next_word = tf.math.argmax(prob, axis=-1)
        next_word = tf.keras.backend.eval(next_word)[0]
        if next_word == S or len(sent) == MAX_LEN:
            break
        sent.append(next_word)
        hist = hist[1:] + [next_word]
    return sent

last_dev = 1e20
best_dev = 1e20
for ITER in range(5):
    # Perform training
    random.shuffle(train)
    # set the model to training mode
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss.numpy()
        train_words += len(sent)
        if (sent_id+1) % 5000 == 0:
            print("--finished %r sentences (word/sec=%.2f)" % (sent_id+1, train_words/(time.time()-start)))
    print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))
  
     # Evaluate on dev set
#     # set the model to evaluation mode
#     model.eval()
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_valid_sent_loss(sent)
        dev_loss += my_loss.numpy()
        dev_words += len(sent)

# Keep track of the development accuracy and reduce the learning rate if it got worse
    if last_dev < dev_loss:
        print("Optimizer lr --->")
        print(optimizer._decayed_lr)
    last_dev = dev_loss

# Keep track of the best development accuracy, and save the model only if it's the best one
    if best_dev > dev_loss:
        model.save_weights('saved_lm_model.h5', save_format='h5', overwrite=True)
        best_dev = dev_loss
  
  # Save the model
    print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words/(time.time()-start)))
  #Generate a few sentences
    for _ in range(5):
        sent = generate_sent()
        print(" ".join([i2w[x] for x in sent]))

