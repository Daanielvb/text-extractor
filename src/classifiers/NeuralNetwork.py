from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, SpatialDropout1D, Conv1D, MaxPool1D
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class NeuralNetwork:

    def __init__(self):
        self.model = None
        pass

    def train(self, X_train, y_train, epochs=150):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=1)

    def cnn_w2v(self, x_train, x_test):
        #TODO: Make this work and check results
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(x_train) + list(x_test))
        x_train_seqs = tokenizer.texts_to_sequences(list(x_train))

        word2idx = tokenizer.word_index

        for word, idx in word2idx.items():
            if word in wv.vocab:
                embeddings[idx] = wv.get_vector(word)

        # Every sequence has the same size. Assigns zero for those who do not arrive at maxlen
        x_train_paded = pad_sequences(x_train_seqs, maxlen=52)
        y_train_onehot = to_categorical(y_train)

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

    def build_LSTM_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        self.model = Sequential()
        embedding_layer = Embedding(vocab_len, 100, weights=[emd_matrix], input_length=long_sent_size,
                                    trainable=True)
        self.model.add(embedding_layer)
        self.model.add(SpatialDropout1D(0.3))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(number_of_classes, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def build_baseline_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        self.model = Sequential()
        embedding_layer = Embedding(vocab_len, 100, weights=[emd_matrix], input_length=long_sent_size,
                                        trainable=True)
        self.model.add(embedding_layer)
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())

        # softmax performing better than relu
        self.model.add(Dense(number_of_classes, activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        return self.model

    def compile_model(self, optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def evaluate_model(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        # TODO: Get loss here to be printed
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores

    def save_model(self, model_name='model.json', weights_name='model.h5'):
        model_json = self.model.to_json()
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(weights_name)
        print("Saved model and weights to disk")

    def load_model(self, model_path='model.json', weights_path='model.h5'):
        # load json and create model
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(weights_path)
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        print("Loaded model from disk")

    def predict_entries(self, entry):
        predictions = self.model.predict_classes(entry)
        # show the inputs and predicted outputs
        print("X=%s, Predicted=%s" % (entry, predictions[0]))
        return predictions
