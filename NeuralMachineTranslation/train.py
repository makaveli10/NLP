from utilities import *
import tensorflow as tf
import numpy as np


MODE = 'train'
URL = 'http://www.manythings.org/anki/fra-eng.zip'
FILENAME = 'fra-eng.zip'
BATCH_SIZE = 64
EMBEDDING_SIZE = 256
RNN_SIZE = 512
NUM_EPOCHS = 15
ATTENTION_FUNC = 'concat'

lines = read_file(URL, filename=FILENAME)
lines = lines.decode('utf-8')

raw_data = []
for line in lines.split('\n'):
    raw_data.append(line.split('\t'))

print(raw_data[-5:])
raw_data = raw_data[:-1]

raw_data_en, raw_data_fr = list(zip(*raw_data))

# pre process input data
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

# tokenize using TensorFlow text pre processing
eng_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
eng_tokenizer.fit_on_texts(raw_data_en)
data_eng = eng_tokenizer.texts_to_sequences(raw_data_en)
data_eng = tf.keras.preprocessing.sequence.pad_sequences(data_eng, padding='post')

print('English Sequences')
print(data_eng[:2])

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')

print('French input sequences')
print(data_fr_in[:2])

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding='post')

print('French output Sequences')
print(data_fr_out[:2])

# Create dataset specifying no of batches
dataset = tf.data.Dataset.from_tensor_slices(
    (data_eng, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(data_eng)).batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.rnn_size = lstm_size
        self.lstm = tf.keras.layers.LSTM(
            self.rnn_size, return_sequences=True, return_state=True)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        output, output_h, output_c = self.lstm(embed, initial_states=state)
        return output, output_h, output_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_size]),
                tf.zeros([batch_size, self.rnn_size]))


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_fn):
        super(LuongAttention, self).__init__()
        self.rnn_size = rnn_size
        self.attention_func = attention_fn

        if self.attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Specified attention function is unknown to the model. Must be dot, general or concat'
            )

        if self.attention_func == 'general':
            self.wa = tf.keras.layers.Dense(self.rnn_size)

        elif self.attention_func == 'concat':
            self.wa = tf.keras.layers.Dense(self.rnn_size, activation='tanh')
            self.va = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # score func : h_t (dot) h_s i.e decoder_output (dot) encoder_output
            # decoder output shape : (batch_size, 1, rnn_size)
            # encoder output shape : (batch_size, max_len, rnn_size)
            # score shape : (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)

        elif self.attention_func == 'general':
            # General score function : dec_out dot (wa dot (enc_out))
            # score shape same : (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)

        elif self.attention_func == 'concat':
            # concat score func : va (dot) tanh(wa (dot) (dec_out + enc_out))
            # Decoder output must be broadcasted to encoder output's shape first
            decoder_output = tf.tile(decoder_output, [1, encoder_output.shape[1], 1])

            # concat -> wa -> va
            # (batch_size, max_len, 2 * rnn_size) -> (batch_size, max_len, rnn_size) -> (batch_size, max_len, 1)

            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))
            # we need to transpose this vector so as to have the same shape (batch_size, 1, max_len)
            score = tf.transpose(score, [0, 2, 1])

        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size, attention_func):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func)
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_states=True)
        self.wa = tf.keras.layers.Dense(self.rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        # decoder input is bach of 1 word sequences
        embed = self.embedding(sequence)

        output, state_h, state_c = self.lstm(embed, initial_state=state)

        # get attention and context
        context, alignment = self.attention(output, encoder_output)

        # combine context and output
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(output, 1)], 1)

        # lstm_out shape (batch_size, 2 * rnn_size)

        # lstm_out has shape (batch_size, rnn_size)
        lstm_out = self.wa(lstm_out)

        # not shape is changed back to vocab space
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment


eng_vocab_size = len(eng_tokenizer.word_index) + 1
encoder = Encoder(eng_vocab_size, EMBEDDING_SIZE, RNN_SIZE)

french_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(french_vocab_size, EMBEDDING_SIZE, RNN_SIZE, ATTENTION_FUNC)

enc_init_states = encoder.init_states(BATCH_SIZE)
enc_outputs = encoder(tf.constant([[1]]), enc_init_states)
dec_outputs = decoder(tf.constant([[1]]), enc_outputs[1:], enc_outputs[0])


def loss_func(targets, logits):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, tf.int64)
    loss = cross_entropy(targets, logits, sample_weight=mask)
    return loss


optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)


def predict(source_text=None):
    if source_text is None:
        source_text = raw_data_en[np.random.choice(len(raw_data_en))]
        source_seq = eng_tokenizer.texts_to_sequences([source_text])

        en_init_states = encoder.init_states(1)
        en_out = encoder(tf.constant(source_seq), en_init_states)

        dec_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
        dec_state_h, dec_state_c = en_out[1:]

        out_words = []
        alignments = []

        while True:
            dec_out, de_state_h, dec_state_c, align = decoder(dec_input, (dec_state_h, dec_state_c), en_out[0])
            dec_input = tf.expand_dims(tf.argmax(dec_out, -1), 0)
            out_words.append(fr_tokenizer.index_word[dec_input.numpy()[0][0]])
            alignments.append(align)

            if out_words[-1] == '<end>' or len(out_words) >= 21:
                break

        print(' '.join(out_words))
        return np.array(alignments), source_text.split(' '), out_words


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        encoder_output = encoder(source_seq, en_initial_states)
        enc_states = encoder_output[1:]
        de_state_h, de_state_c = enc_states

        for i in range(target_seq_out.shape[1]):
            # decoder input shape : (batch_size,length)
            decoder_input = tf.expand_dims(target_seq_in[:, i], 1)
            logits, de_state_h, de_state_c, _ = decoder(
                decoder_input, (de_state_h, de_state_h), enc_outputs[0])

            loss += loss_func(target_seq_out[:, i], logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]


if not os.path.exists('checkpoints/encoder'):
    os.makedirs('checkpoints/encoder')

if not os.path.exists('checkpoints/decoder'):
    os.makedirs('checkpoint/decoder')


# uncomment for inference mode
enc_checkpoint = tf.train.latest_checkpoint('checkpoints/encoder')
dec_checkpoint = tf.train.latest_checkpoint('checkpoints/decoder')


if enc_checkpoint is not None and dec_checkpoint is not None:
    encoder.load_weights(enc_checkpoint)
    decoder.load_weights(dec_checkpoint)

if MODE == 'train':
    for e in range(NUM_EPOCHS):
        en_initial_states = encoder.init_states(BATCH_SIZE)
        encoder.save_weights(
            'checkpoints_luong/encoder/encoder_{}.h5'.format(e + 1))
        decoder.save_weights(
            'checkpoints_luong/decoder/decoder_{}.h5'.format(e + 1))
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
            if batch % 100 == 0:
                print("Epoch {}, Batch {}, Loss {}").format(e+1, batch, loss.numpy())

        try:
            predict()
            predict('How are you today?')
        except Exception:
            continue



