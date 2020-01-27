import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class ModelUtil:

    def __init__(self):
        pass

    @staticmethod
    def decode_labels(label_encoder, encoded_array):
        return label_encoder.inverse_transform(np.argmax(encoded_array, axis=1, out=None))

    @staticmethod
    def decode_class(label_encoder, class_value):
        return label_encoder.inverse_transform([class_value])

    @staticmethod
    def save_encoder(label_encoder, encoder_file_name='label_encoder_classes.npy'):
        np.save(encoder_file_name, label_encoder.classes_)

    @staticmethod
    def load_encoder(encoder_file_name='label_encoder_classes.npy'):
        encoder = LabelEncoder()
        encoder.classes_ = np.load(encoder_file_name, allow_pickle=True)
        return encoder

    @staticmethod
    def save_tokenizer(tokenizer, tokenizer_file='tokenizer.pickle'):
        with open(tokenizer_file, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_tokenizer(tokenizer_file='tokenizer.pickle'):
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    @staticmethod
    def normalize_data(data, norm='l2'):
        """ normalizes input data
        https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
        return preprocessing.normalize(data, norm)

    @staticmethod
    def remove_entries_based_on_threshold(dataframe, class_name, threshold):
        """Remove classes from a dataframe based on threshold, eg: If threshold = 1,
            only instances with 2 or more examples contained in the `class_name` column
            will remain on the dataframe
        """
        # TODO: Check some NaNs
        return dataframe.groupby(class_name).filter(lambda x: len(x) > threshold)

    @staticmethod
    def normalize_dataframe(csv_file='../../data/parsed-data/stylo.csv', class_name='Classe(Autor)', threshold=2):
        df = pd.read_csv(csv_file)
        df = ModelUtil().remove_entries_based_on_threshold(df, class_name, threshold)
        y = df.pop(class_name)
        columns = df.columns
        normalized_data = ModelUtil.normalize_data(df.values)
        df = pd.DataFrame(columns=columns, data=normalized_data)
        y = y.reset_index()
        df[class_name] = y['Classe(Autor)']
        return df

    @staticmethod
    def extract_validation_data_from_frame(X, Y):
        # Get one example of each class for validation:
        validation_data = {}
        indexes = []
        for idx, author in enumerate(Y):
            if author not in validation_data.keys():
                print(author)
                validation_data[author] = X.iloc[idx]
                indexes.append(idx)

        bias = 0
        for idx in indexes:
            Y.drop(Y.index[idx + bias], inplace=True)
            X.drop(X.index[idx + bias], inplace=True)
            bias -= 1
        return validation_data, X, Y

    @staticmethod
    def extract_validation_data(X, Y):
        # Get one example of each class for validation:
        validation_data = {}
        indexes = []
        for idx, author in enumerate(reversed(Y)):
            if author not in validation_data.keys():
                print(author)
                validation_data[author] = X[idx]
                indexes.append(idx)

        for idx in reversed(indexes):
            Y = np.delete(Y, idx)
            X = np.delete(X, idx, axis=0)

        return validation_data, X, Y
