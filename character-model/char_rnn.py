import pickle, gzip
import numpy as np
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Embedding, Input, GRU
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers.wrappers import Bidirectional

import os
import random

EMB_DIM = 150
# lines = pickle.load(gzip.open("all_data2.gzip", 'rb'))[:100]
# raw_text = ' '.join(lines)

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# dump(mapping, open('mapping.pkl', 'wb'))
mapping = pickle.load(open('mapping.pkl', 'rb'))
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

embedding_matrix = np.zeros((vocab_size, EMB_DIM))
for char, i in mapping.items():
    # embedding_vec =
    embedding_matrix[i] = np.array([random.random() for i in range(embedding_matrix.shape[1])])

# separate into input and output
sequences = array(sequences)
print(sequences.shape)
X, y = sequences[:,:-1], sequences[:,-1]
# sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)


def write_weights_to_file(embedding_weights, mapping):
    filename = 'char2vec.txt'
    char2vec = {}
    for idx, weights in enumerate(embedding_weights):
        char2vec[list(mapping.keys())[list(mapping.values()).index(idx)]] = weights

    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in char2vec.items():
                file.write('%s' % key)
                for each in value:
                    file.write(' %s' % each)
                file.write('\n')

def create_model(X_max_len):
    char_in = Input(shape=(X_max_len,), name='input')
    emb_layer = Embedding(vocab_size, EMB_DIM, weights=[embedding_matrix],
                    input_length=X_max_len, trainable=True, name='emb')(char_in)
    lstm1 = Bidirectional(GRU(75, return_sequences=True, name='lstm1'))(emb_layer)
    lstm2 = Bidirectional(GRU(75, name='lstm2'))(lstm1)
    drop = Dropout(0.5)(lstm2)
    fc = Dense(vocab_size, activation='softmax', name='dense')(drop)

    model = Model(input=char_in, output=fc)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model(X.shape[1])
checkpointer = ModelCheckpoint(filepath="model.hdf5", verbose=1, save_best_only=True)
early_stop = EarlyStopping(patience=10)
# fit model
# hist = model.fit(X, y, epochs=500, verbose=1, validation_split=0.2, callbacks=[early_stop,checkpointer])

model.load_weights("model.hdf5")
embedding_weights = model.get_layer('emb').get_weights()[0]

print(embedding_weights)
write_weights_to_file(embedding_weights, mapping)

