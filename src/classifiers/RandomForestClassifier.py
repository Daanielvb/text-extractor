import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.util.ModelUtil import *
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

class RandomForestClassifier:

    def __init__(self):
        self.X = None
        self.Y = None
        self.le = None
        pass

    def pre_process(self, df):
        # show_column_distribution(df, 'Author')

        self.Y = df.pop('Author')
        self.X = df

        self.le = LabelEncoder()
        self.le.fit(self.Y)
        encoded_Y = self.le.transform(self.Y)
        ModelUtil.save_encoder(self.le)
        # decode: le.inverse_transform(encoded_Y)

    def split_and_train(self, df):
        rfc = self.build_classifiers()

        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
        for train_index, test_index in kfold.split(self.X, self.Y):
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_y = np_utils.to_categorical(self.Y)
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = dummy_y[train_index], dummy_y[test_index]
            rfc.fit(X_train, X_test)

    def build_classifier(self):
        return RandomForestClassifier()
