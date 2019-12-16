import tensorflow as tf


class tfBoW(tf.keras.Model):
    def __init__(self, n_words, n_tags):
        super(tfBoW, self).__init__()
        self.bias = tf.Variable(tf.zeros([n_words, n_tags]),trainable=True, name='bias')
        self.embedding = tf.keras.layers.Embedding(n_words, n_tags, embeddings_initializer='glorot_uniform')

    def call(self, words):
        embed_out = self.embedding(words)
        out = tf.math.reduce_sum(embed_out, axis=0) + self.bias
        out = tf.reshape(out, [1, -1])
        return out


class tfCBow(tf.keras.Model):
    def __init__(self, nwords, ntags, emb_size):
        super(tfCBow, self).__init__()
        self.embedding = tf.keras.layers.Embedding(nwords,
                             emb_size, 
                             )
        self.linear = tf.keras.layers.Dense(ntags, 
                        use_bias=True, 
                        kernel_intializer='glorot_uniform')
    
    def call(self, words):
        emb_out = self.embedidng(words)
        emb_sum = tf.math.reduce_sum(emb_out, axis=0)
        emb_out = tf.reshape(emb_out, [1, -1])
        out = self.linear(emb_out)
        return emb_out


class tfDeepCBow(tf.keras.Model):
    def __init__(self, nwords, ntags, emb_size, nlayers, hid_size):
        super(tfDeepCBow, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            nwords,
            emb_size,
            embeddings_initializer='glorot_uniform'
        )
        self.linears = []
        for i in range(nlayers):
            if i == 0:
                dense = tf.keras.layers.Dense(
                    emb_size,
                    activation='tanh',
                    use_bias=True, 
                    kernel_intializer='glorot_uniform')
            else:    
                dense = tf.keras.layers.Dense(
                    hid_size,
                    activation='tanh',
                    use_bias=True, 
                    kernel_intializer='glorot_uniform')
            self.linears.append(dense)
        self.output_layer = tf.keras.layers.Dense(
                        hid_size,
                        use_bias=True, 
                        kernel_intializer='glorot_uniform')
        
    def call(self, words):
        emb_out = self.embedding(words)
        out = tf.math.reduce_sum(emb_out)
        out = tf.reshape(out, [1, -1])
        for linear in self.linears:
            out = linear(out)
        out = self.output_layer(out)
        return out
            
                
                                            