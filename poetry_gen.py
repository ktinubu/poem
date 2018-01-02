''' generate poetry output from seed. model trained on reddit poetry.

sourrces: 
    https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
    https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
'''

from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class FullModel:
    def __init__(self):
        (
            self.model,
            self.maxlen,
            self.chars, 
            self.char_indices, 
            self.indices_char
        ) = get_model_inititializer()

    def predict(self, sentence):
        sentence = pad_sentence(sentence).lower()
        if not sentence.isalnum:
            print("input must be alpha numberic")
            return

        diversity = 1.0 # values can vary through [0.2, 0.5, 1.0, 1.2]:
        generated = ' '
        # print()
        # print('----- diversity:', diversity)
        # print('----- Generating with seed: "' + sentence + '"')

        for i in range(100):
            x_pred = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        return generated.strip().strip('\n')

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

def get_model_inititializer():
    text = io.open("redditPoems.txt").read().lower()
    #print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    #print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # load weights
    model.load_weights("improved weights-00-1.4095.hdf5")

    return (model, maxlen, chars, char_indices, indices_char)

# currently model only works if input is 40 chars longs so must pad/slice
def pad_sentence(sentence):
    if len(sentence) > 40:
        return sentence[:41]
    while len(sentence) != 40:
        sentence = sentence + " "
    #print(len(sentence))
    return sentence

if __name__ == '__main__':
    sentence = "who are you even"
    full_model = FullModel()
    generated = full_model.predict(sentence)
    print(generated)
