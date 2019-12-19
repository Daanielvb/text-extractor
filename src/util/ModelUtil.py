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
        df = ModelUtil().remove_entries_based_on_threshold(df, 'Classe(Autor)', threshold)
        y = df.pop(class_name)
        columns = df.columns
        normalized_data = ModelUtil.normalize_data(df)
        df = pd.DataFrame(columns=columns, data=normalized_data)
        df[class_name] = y
        return df
