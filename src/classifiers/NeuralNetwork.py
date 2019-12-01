from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import model_from_json


class NeuralNetwork:

    def __init__(self):
        self.model = None
        pass

    def train(self, X_train, y_train, epochs=150):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=1)

    def build_baseline_model(self, emd_matrix, long_sent_size, vocab_len, number_of_classes):
        self.model = Sequential()
        embedding_layer = Embedding(vocab_len, 100, weights=[emd_matrix], input_length=long_sent_size,
                                        trainable=False)
        self.model.add(embedding_layer)
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

    def predict_entry(self, entry):
        return self.model.evaluate(entry)
