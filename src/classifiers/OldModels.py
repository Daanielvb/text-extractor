from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Activation, Conv1D, MaxPool1D, Dense, Flatten, Dropout, GRU, Input, Concatenate, GlobalMaxPooling1D

import numpy as np


def cnn(word2idx):
    embeddings = np.zeros((len(word2idx) + 1, 100))
    # Approach without word2vec
    cnn_model = Sequential()

    cnn_model.add(Embedding(embeddings.shape[0],
                            embeddings.shape[1],
                            trainable=True, input_length=52))

    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv1D(64, 5, activation='relu'))
    cnn_model.add(MaxPool1D(2, strides=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(5, activation='softmax'))
    return cnn_model


def cnn_w2v(word2idx):
    embeddings = np.zeros((len(word2idx) + 1, 100))
    # Approach with word2vec
    cnn_model = Sequential()

    cnn_model.add(Embedding(embeddings.shape[0],
                            embeddings.shape[1],
                            weights=[embeddings],
                            trainable=False, input_length=52))
    # Prevents overfitting
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv1D(64, 5, activation='relu'))
    # Get the most relevant features
    cnn_model.add(MaxPool1D(2, strides=2))
    # Transforms the input data to calculate the density
    cnn_model.add(Flatten())
    cnn_model.add(Dense(5, activation='softmax'))
    return cnn_model

def lstm_trainable(dic_size, trainable):
    # LSTM Model without trainable
    lstm_model = Sequential()
    lstm_model.add(Embedding(dic_size, 128))
    lstm_model.add(LSTM(128, trainable=trainable, dropout=0.2))
    lstm_model.add(Dense(5))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Activation('softmax'))
    return lstm_model


def gru(embeddings):
    gru_model = Sequential()
    gru_model.add(Embedding(embeddings.shape[0],
                            100,
                            trainable=True))
    gru_model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    gru_model.add(Dense(5, activation='softmax'))
    return gru_model