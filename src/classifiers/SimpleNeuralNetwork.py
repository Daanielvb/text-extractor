from src.util.ModelUtil import *
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sklearn.metrics
from keras.utils import np_utils


class SimpleNeuralNetwork:

    def __init__(self, df):
        self.X = None
        self.Y = None
        self.encoded_Y = None
        self.le = None
        self.model = None
        self.pre_process(df)
        pass

    def pre_process(self, df):
        # show_column_distribution(df, 'Author')

        self.Y = df.pop('Classe(Autor)')
        self.X = df
        self.le = LabelEncoder()
        self.le.fit(self.Y)
        self.encoded_Y = self.le.transform(self.Y)
        ModelUtil.save_encoder(self.le)
        # decode: le.inverse_transform(encoded_Y)

    def split_and_train(self):
        self.build_classifier()
        cv_scores = []
        models = []
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)

        for train_index, test_index in kfold.split(self.X, self.Y):
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_y = np_utils.to_categorical(self.encoded_Y)
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = dummy_y[train_index], dummy_y[test_index]
            self.model.fit(X_train, y_train, epochs=300, verbose=1)
            scores = self.evaluate_model(X_test, y_test)
            cv_scores.append(scores[1] * 100)
            models.append(self.model)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
        #models[cv_scores.index(max(cv_scores))].save_model()

    def build_classifier(self, number_of_features=52, number_of_classes=11):
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=number_of_features, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(number_of_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def evaluate_model(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        # TODO: Get loss here to be printed
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores

