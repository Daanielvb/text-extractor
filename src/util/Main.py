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
import pickle

def convert_data(base_path):
    # TODO: Re-add lygya maria souza and maria milena dos santos at MECB/A3
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


def remove_entries_based_on_threshold(dataframe, class_name, threshold):
    """Remove classes from a dataframe based on threshold, eg: If threshold = 1,
        only instances with 2 or more examples contained in the `class_name` column
        will remain on the dataframe
    """
    #TODO: Check some NaNs
    return dataframe.groupby(class_name).filter(lambda x: len(x) > threshold)


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
    return label_encoder.inverse_transform([class_value])


def save_encoder(label_encoder, encoder_file_name='label_encoder_classes.npy'):
    np.save(encoder_file_name, label_encoder.classes_)


def load_encoder(encoder_file_name='label_encoder_classes.npy'):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_file_name, allow_pickle=True)
    return encoder


def save_tokenizer(tokenizer, tokenizer_file='tokenizer.pickle'):
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(tokenizer_file='tokenizer.pickle'):
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def run_compiled_model(model, tokenizer, encoder, X_predict, y_expected):
    embedded_sentence = tokenizer.texts_to_sequences([X_predict])
    padded_sentences = pad_sequences(embedded_sentence, 6658, padding='post')

    pred = model.predict_entries(padded_sentences)

    decoded_pred = decode_class(encoder, pred[0])
    print('predicted:' + decoded_pred + ' expected:' + y_expected)
    return decoded_pred == y_expected


def run_complete_pipeline():
    df = pd.read_csv('../../data/parsed-data/data2.csv')

    df = remove_entries_based_on_threshold(df, 'Author', 1)
    y = df.pop('Author')

    le = LabelEncoder()
    le.fit(y)
    encoded_Y = le.transform(y)
    save_encoder(le)
    # decode: le.inverse_transform(encoded_Y)

    tokenizer, padded_sentences, max_sentence_len \
        = PortugueseTextualProcessing().convert_corpus_to_number(df)

    # save_tokenizer(tokenizer)
    vocab_len = len(tokenizer.word_index) + 1

    glove_embedding = PortugueseTextualProcessing().load_vector_2(tokenizer)

    embedded_matrix = PortugueseTextualProcessing().build_embedding_matrix(glove_embedding, vocab_len, tokenizer)

    # TODO: Iterate over params to check best configs
    init = ['glorot_uniform', 'normal', 'uniform']
    optimizers = ['rmsprop', 'adam']
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    param_network = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)

    # TODO: Check results with normalization (df_norm = (df - df.mean()) / (df.max() - df.min()))
    cv_scores = []
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
    models = []

    # Separate some validation samples

    for train_index, test_index in kfold.split(padded_sentences, encoded_Y):
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = padded_sentences[train_index], padded_sentences[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]

        nn = NeuralNetwork()
        nn.build_baseline_model(embedded_matrix, max_sentence_len, vocab_len, len(dummy_y[0]))

        nn.train(X_train, y_train, 100)

        scores = nn.evaluate_model(X_test, y_test)
        cv_scores.append(scores[1] * 100)
        models.append(nn)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    models[cv_scores.index(max(cv_scores))].save_model()


if __name__ == '__main__':
    df = pd.read_csv('../../data/parsed-data/data2.csv')

    from random import randint

    correct = 0
    df = remove_entries_based_on_threshold(df, 'Author', 1)
    nn = NeuralNetwork()
    nn.load_model()
    encoder = load_encoder()
    tokenizer = load_tokenizer()
    for i in range(100):
        idx = randint(0, len(df.Author) - 1)
        print('idx:' + str(idx))
        half_size = int(len(df.Text.iloc[idx])/2)
        half_text = df.Text.iloc[idx][half_size:]
        pred_result = run_compiled_model(nn, tokenizer, encoder, half_text, df.Author.iloc[idx])
        if pred_result:
            correct += 1

    print('total correct = ' + str(correct))
    print('accuracy % = ' + str((correct/100) * 100))
