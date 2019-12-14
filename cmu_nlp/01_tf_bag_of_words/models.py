import tensorflow as tf


class tfBoW(tf.keras.Model):
    def __init__(self, n_words, n_tags):
        super(tfBoW, self).__init__()
        self.bias = tf.Variable(tf.zeros([n_words, n_tags]),trainable=True, name='bias')
        self.embedding = tf.keras.layers.Embedding(n_words,
                                                   n_tags,
                                                   embeddings_initializer='uniform')

    def call(self, words):
        embed_out = self.embedding(words)
        out = tf.keras.backend.sum(embed_out, axis=0) + self.bias
        out = tf.reshape(out, [1, -1])
        return out
