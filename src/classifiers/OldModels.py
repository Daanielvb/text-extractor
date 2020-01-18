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


def cnn_gru(word2idx):
    embeddings = np.zeros((len(word2idx) + 1, 100))
    # CNN/GRU Model
    inputs = Input(shape=(52,), name='input')

    embedding = Embedding(embeddings.shape[0], embeddings.shape[1], trainable=True, input_length=52)(inputs)

    # CNN uni,bi e tri grama
    conv1 = Conv1D(100, 1, activation='relu')(embedding)
    conv2 = Conv1D(100, 3, activation='relu')(embedding)
    conv3 = Conv1D(100, 2, activation='relu')(embedding)

    conv1 = GlobalMaxPooling1D()(conv1)
    conv2 = GlobalMaxPooling1D()(conv2)
    conv3 = GlobalMaxPooling1D()(conv3)

    # Concatenate CNNs results
    concatenate = Concatenate()([conv1, conv2, conv3])

    dense1 = Dense(100, activation='tanh')(concatenate)
    dense1 = Dropout(0.2)(dense1)

    # GRU model
    gru = GRU(100, dropout=0.2, recurrent_dropout=0.2)(embedding)

    # Concatenate GRU and CNNs
    merge = Concatenate()([dense1, gru])

    out1 = Dense(128, activation='tanh')(merge)
    out = Dense(5, activation='softmax')(out1)

    model = Model(inputs=inputs, outputs=out)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

def lstm_trainable(dic_size, trainable):
    # LSTM Model without trainable
    lstm_model = Sequential()
    lstm_model.add(Embedding(dic_size, 128))
    lstm_model.add(LSTM(128, trainable=trainable, dropout=0.2))
    lstm_model.add(Dense(5))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Activation('softmax'))