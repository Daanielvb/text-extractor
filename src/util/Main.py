from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from src.classifiers.NeuralNetwork import *


def convert_data(base_path):
    clean_txt_content, txt_file_names = FileUtil.convert_files(base_path)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_paths = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    CSVReader.write_files('../../data/parsed-data/', file_paths, 'data2.csv', files_content)


def prepare_train_data(dataframe):
    Meta_Y = dataframe.pop('Autor')
    Meta_X = dataframe
    X_train, X_test, y_train, y_test = train_test_split(Meta_X, Meta_Y, test_size=0.4, stratify=Meta_Y)
    return X_train, X_test, y_train, y_test


def run_svc(X_train, X_test, y_train, y_test):
    clf = SVC(gamma='scale')
    clf.fit(X_train, y_train)
    print(clf.predict(X_train))
    print(clf.score(X_test, y_test))


def remove_group_works(dataframe):
    dataframe['Author'] = dataframe['Author'].astype('str')
    mask = (dataframe['Author'].str.len() <= 5)
    return dataframe.loc[mask]


def remove_single_class_entries(dataframe, class_name):
    #TODO: Check some NaNs
    return dataframe.groupby(class_name).filter(lambda x: len(x) > 1)


def save_plot(plot, plot_name):
    plot.savefig(plot_name)


def show_column_distribution(dataframe, class_name):
    dataframe[class_name].value_counts().plot(kind='barh')
    plt.show()


def save_converted_stylo_data():
    CSVReader().write_stylo_features('../../data/parsed-data/', 'stylo.csv', CSVReader.read_csv('../../data/parsed-data/data2.csv'))


def normalize_data(data, norm='l2'):
    """ normalizes input data
    https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
    return preprocessing.normalize(data, norm)


def decode_labels(label_encoder, encoded_array):
    return label_encoder.inverse_transform(np.argmax(encoded_array, axis=1, out=None))


def decode_class(label_encoder, class_value):
    label_encoder.inverse_transform([class_value])


if __name__ == '__main__':

    df = pd.read_csv('../../data/parsed-data/data2.csv')

    df = remove_single_class_entries(df, 'Author')
    y = df.pop('Author')

    le = LabelEncoder()
    le.fit(y)
    encoded_Y = le.transform(y)

    # decode: le.inverse_transform(encoded_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    result = []
    glove_embedding = PortugueseTextualProcessing().load_vector_2()

    tokenizer, padded_sentences, max_sentence_len \
        = PortugueseTextualProcessing().convert_corpus_to_number(df)

    vocab_len = len(tokenizer.word_index) + 1

    embedded_matrix = PortugueseTextualProcessing().build_embedding_matrix(glove_embedding, vocab_len, tokenizer)

    # TODO: Iterate over params to check best configs
    init = ['glorot_uniform', 'normal', 'uniform']
    optimizers = ['rmsprop', 'adam']
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    param_network = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

    # TODO: Check results with normalization (df_norm = (df - df.mean()) / (df.max() - df.min()))

    number_of_classes = len(np_utils.to_categorical(encoded_Y)[0])

    cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    models = []
    for train_index, test_index in kfold.split(padded_sentences, encoded_Y):
        dummy_y = np_utils.to_categorical(encoded_Y)
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = padded_sentences[train_index], padded_sentences[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]

        nn = NeuralNetwork()
        model = nn.baseline_model(embedded_matrix, max_sentence_len, vocab_len, len(encoded_Y[0]))

        nn.train(X_train, y_train, 100)

        scores = nn.evaluate_model(X_test, y_test)
        cv_scores.append(scores[1] * 100)
        models.append(nn)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    models[cv_scores.index(max(cv_scores))].save_model()







