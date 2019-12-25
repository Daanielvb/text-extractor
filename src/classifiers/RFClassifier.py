import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.util.ModelUtil import *
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from sklearn import datasets, metrics, model_selection, svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:

    def __init__(self, df):
        self.X = None
        self.Y = None
        self.encoded_Y = None
        self.le = None
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
        rfc = self.build_classifier()

        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=11)
        scores, precision, f1 = [], [], [], [], [], [], []

        for train_index, test_index in kfold.split(self.X, self.Y):
            # convert integers to dummy variables (i.e. one hot encoded)
            # dummy_y = np_utils.to_categorical(self.Y)
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            rfc.fit(X_train, y_train)

            y_pred = rfc.predict(X_test)
            precision.append(metrics.precision_score(y_test, y_pred, average='weighted'))
            f1.append(metrics.f1_score(y_test, y_pred, average='weighted'))
            scores.append(rfc.score(X_test, y_test))

        print(scores)

    def build_classifier(self):
        clf = RandomForestClassifier(
            n_estimators=50,
            criterion='gini',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=0,
            verbose=0,
            warm_start=False,
            class_weight='balanced'
        )
        return clf
