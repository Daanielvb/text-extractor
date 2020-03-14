from src.util.ModelUtil import *
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import sklearn.metrics
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SimpleNeuralNetwork:
    """Neural network used for stylometric features with some improvements after the original NeuralNetwork"""

    def __init__(self, df):
        self.X = None
        self.Y = None
        self.encoded_Y = None
        self.le = None
        self.model = None
        self.pre_process(df)

    def pre_process(self, df):
        # show_column_distribution(df, 'Author')
        self.Y = df.pop('Author')
        self.X = df
        self.X_val = None
        self.y_val = None
        scaled_features = PowerTransformer().fit_transform(self.X.values)
        self.X = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        self.le = LabelEncoder()
        self.le.fit(self.Y)
        self.encoded_Y = self.le.transform(self.Y)
        #self.save_validation_data()
        ModelUtil.save_encoder(self.le)
        # decode: le.inverse_transform(encoded_Y)

    def save_validation_data(self):
        self.X, self.X_val, self.Y, self.y_val = train_test_split(self.X, self.Y, random_state=7, test_size=0.15)

    def simple_split_train(self):
        self.build_classifier(number_of_features=len(self.X.columns), number_of_classes=len(set(self.encoded_Y)))

        X, X_test, Y, y_test = train_test_split(self.X, self.Y, random_state=7, test_size=0.25)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=7, test_size=0.5)

        self.model.fit(X, np_utils.to_categorical(self.le.transform(Y)), epochs=250, verbose=0)

        y_pred = self.le.inverse_transform(self.model.predict_classes(X_test))
        score = accuracy_score(y_test, y_pred)
        print("Test score: %.2f%%" % (score * 100))
        return score

        # y_pred = self.le.inverse_transform(self.model.predict_classes(X_val))
        # score = accuracy_score(y_val, y_pred)
        # print("Validation score: %.2f%%" % (score * 100))
        # return score

    def split_and_cross_validate(self):
        self.build_classifier(number_of_features=len(self.X.columns), number_of_classes=len(set(self.encoded_Y)))
        cv_scores = []
        models = []
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

        for train_index, test_index in kfold.split(self.X, self.Y):
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_y = np_utils.to_categorical(self.encoded_Y)
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = dummy_y[train_index], dummy_y[test_index]
            self.model.fit(X_train, y_train, epochs=300, verbose=1)
            scores = self.evaluate_model(X_test, y_test)
            cv_scores.append(scores[1] * 100)
            models.append(self.model)

        print("Train score: %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
        return np.mean(cv_scores)

    def build_classifier(self, number_of_features, number_of_classes=11):
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=number_of_features, activation='relu'))
        self.model.add(Dropout(0.3))
        # self.model.add(Dense(16, activation='relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(number_of_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def evaluate_model(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        # TODO: Get loss here to be printed
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores

