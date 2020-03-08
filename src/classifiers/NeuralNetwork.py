from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, SpatialDropout1D, Conv1D, MaxPool1D, GRU, Input, Concatenate, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import model_from_json, Model


class NeuralNetwork:

    def __init__(self):
        self.model = None
        pass

    def train(self, X_train, y_train, epochs=150):
        return self.model.fit(X_train, y_train, epochs=epochs, verbose=1)

    def build_CNN_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        self.model = Sequential()
        self.model.add(Embedding(vocab_len,
                                100,
                                weights=[emd_matrix],
                                trainable=False, input_length=long_sent_size))
        # Prevents overfit
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(64, 5, activation='relu'))
        # Get the most relevant features
        self.model.add(MaxPool1D(2, strides=2))
        # Transforms the input data to calculate the density
        self.model.add(Flatten())
        self.model.add(Dense(number_of_classes, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
        return self.model

    def build_LSTM_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        self.model = Sequential()
        embedding_layer = Embedding(vocab_len, 100, weights=[emd_matrix], input_length=long_sent_size,
                                    trainable=False)
        self.model.add(embedding_layer)
        self.model.add(SpatialDropout1D(0.3))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(number_of_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def build_cnn_gru_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        inputs = Input(shape=(long_sent_size,), name='input')

        embedding = Embedding(vocab_len, 100, weights=[emd_matrix], trainable=True, input_length=long_sent_size)(inputs)

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
        out = Dense(number_of_classes, activation='softmax')(out1)

        self.model = Model(inputs=inputs, outputs=out)

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return self.model

    def build_baseline_model(self, emd_matrix, longest_sent_size, vocab_size,
                             number_of_classes, emb_size=100):
        self.model = Sequential()
        embedding_layer = Embedding(vocab_size, emb_size, weights=[emd_matrix],
                                    input_length=longest_sent_size, trainable=False)
        self.model.add(embedding_layer)
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(number_of_classes, activation='softmax'))

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

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
