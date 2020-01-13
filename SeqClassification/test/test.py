import tensorflow as tf
import pickle
import sys
import numpy as np


MAX_SEQ_LEN = 120


id2emotion = {}
for line in open('test/id2emotion.map'):
    k, v = line.strip().split('\t')
    id2emotion[int(k)] = v


# Load tokenizer
with open('test/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
classifier = tf.keras.models.load_model('test/classifier.h5')
print("Classifier Loaded")


def predict(inp_seq):
    # convert inp_seq to padded tokens
    seqs = [inp_seq]

    test_data = tokenizer.texts_to_sequences(seqs)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, 
                                                              padding='post', 
                                                              maxlen=MAX_SEQ_LEN)
    input_arr = np.reshape(test_data, (1, MAX_SEQ_LEN))
    prediction = classifier.predict(input_arr)
    print(prediction)
    b = tf.math.argmax(prediction[0])
    c = tf.keras.backend.eval(b)

    return id2emotion[c]


def main():
    seq = sys.argv[1]
    if not isinstance(seq, str):
        print("Please entring string object")
    print(predict(sys.argv[1]))

          
if __name__ == '__main__':
    main()