from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
from src.util.ModelUtil import *
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


def save_plot(plot_name):
    plt.savefig(plot_name)


def show_column_distribution(dataframe, class_name):
    dataframe[class_name].value_counts().plot(kind='barh')
    plt.show()


def save_converted_stylo_data():
    CSVReader().write_stylo_features('../../data/parsed-data/', 'stylo.csv', CSVReader.read_csv('../../data/parsed-data/data2.csv'))


def normalize_data(data, norm='l2'):
    """ normalizes input data
    https://scikit-learn.org/stable/modules/preprocessing.html#normalization"""
    return preprocessing.normalize(data, norm)

def run_compiled_model(model, tokenizer, encoder, X_predict, y_expected):
    embedded_sentence = tokenizer.texts_to_sequences([X_predict])
    padded_sentences = pad_sequences(embedded_sentence, 6658, padding='post')

    pred = model.predict_entries(padded_sentences)

    decoded_pred = ModelUtil.decode_class(encoder, pred[0])
    print('predicted:' + decoded_pred + ' expected:' + y_expected)
    return decoded_pred == y_expected


def run_compiled_pipeline():
    df = pd.read_csv('../../data/parsed-data/data2.csv')
    from random import randint
    # TODO: Check if null tokens is a good approach for unseen content
    # TODO: Perform validation with unseen data
    # TODO: See if its possible to use Random Forest with our embedding matrix
    # TODO: Think about the group works problem
    correct = 0
    df = remove_entries_based_on_threshold(df, 'Author', 1)
    nn = NeuralNetwork()
    nn.load_model()
    encoder = ModelUtil().load_encoder()
    tokenizer = ModelUtil().load_tokenizer()
    run_compiled_model(nn, tokenizer, encoder,
                       '1 - Competição entre membros de espécies diferentes que usam os mesmos recursos limitados. Os recursos costumam ser limitados em um habitat e muitas espécies podem competir para consegui-los. O efeito geral da competição interespecífica é negativo para as duas espécies participantes, ou seja, cada espécie estaria melhor se a outra espécie não estivesse presente. Com tudo referente à complementaridade de nicho O princípio exclusão competitiva diz que duas espécies competidoras podem concorrer em determinado local, mas para isso elas precisam possuir nichos realizados diferentes. 2 Em referente ao primeiro estudo de caso (Asterionella formosa/ Synedra ulna) a ocorrência competitiva devido ao mesmo recurso (silicato) apresenta princípio da exclusão competitivas onde ambas ocupando o mesmo nicho em que a capacidade suporte influencia na exclusão de uma espécie. No segundo caso a relação de coexistência da diversidade de espécie de peixe-palhaço esta relacionada com a quantidade de anêmonas onde o seu principal recursos esta no abrigo possibilitando a produtividade e perpetuação da espécie de peixe, além disso, devido a população desse peixe esta ligado ao recurso limitante por anêmona criando uma diversidade que utiliza diferentes nichos devido ao distanciamento de cada anêmona do estudo. 3 - O princípio da exclusão competitiva ou, como também é chamado, Lei de Gause, é uma proposição que afirma que, em um ambiente estável no qual os indivíduos se distribuem de forma homogênea, duas espécies com nichos ecológicos parecidos não podem coexistir, devido a pressão evolutiva exercida pela competição. De acordo com esse princípio, um dos competidores terminará por sobrepujar ao outro, o que pode acarretar mudanças morfológicas, comportamentais, deslocamento de nicho ecológico ou até mesmo a extinção da espécie em desvantagem. Em suma, o que esse conceito quer dizer é que competidores completos não podem coexistir. ',
                       'test')

    for i in range(100):
        idx = randint(0, len(df.Author) - 1)
        print('idx:' + str(idx))
        half_size = int(len(df.Text.iloc[idx]) / 2)
        half_text = df.Text.iloc[idx][:half_size]
        pred_result = run_compiled_model(nn, tokenizer, encoder, half_text, df.Author.iloc[idx])
        if pred_result:
            correct += 1

    print('total correct = ' + str(correct))
    print('accuracy % = ' + str((correct / 100) * 100))


def run_complete_pipeline():
    convert_data('../../data/students_exercises')
    df = pd.read_csv('../../data/parsed-data/data2.csv')

    df = remove_entries_based_on_threshold(df, 'Author', 1)

    #show_column_distribution(df, 'Author')

    y = df.pop('Author')

    le = LabelEncoder()
    le.fit(y)
    encoded_Y = le.transform(y)
    ModelUtil().save_encoder(le)
    # decode: le.inverse_transform(encoded_Y)

    tokenizer, padded_sentences, max_sentence_len \
        = PortugueseTextualProcessing().convert_corpus_to_number(df)

    ModelUtil().save_tokenizer(tokenizer)
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

    nn = NeuralNetwork()
    nn.build_baseline_model(embedded_matrix, max_sentence_len, vocab_len, len(np_utils.to_categorical(encoded_Y)[0]))

    for train_index, test_index in kfold.split(padded_sentences, encoded_Y):
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = padded_sentences[train_index], padded_sentences[test_index]
        y_train, y_test = dummy_y[train_index], dummy_y[test_index]
        nn.train(X_train, y_train, 100)

        scores = nn.evaluate_model(X_test, y_test)
        cv_scores.append(scores[1] * 100)
        models.append(nn)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
    models[cv_scores.index(max(cv_scores))].save_model()


if __name__ == '__main__':
    #CSVReader().read_csv('../../data/parsed-data/data2.csv')
    #save_converted_stylo_data()
    df = pd.read_csv('../../data/parsed-data/stylo2.csv')

    df.to_csv('../../data/parsed-data/stylo2.csv')
    #TODO : Create random forest class and move utility methods to other places outside main