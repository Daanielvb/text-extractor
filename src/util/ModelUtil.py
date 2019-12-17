import numpy as np
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