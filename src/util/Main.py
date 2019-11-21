from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing


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


if __name__ == '__main__':

    #word_embedding = PortugueseTextualProcessing().load_vector_2()

    #embedding_matrix = PortugueseTextualProcessing().build_embedding_matrix(word_embedding, ['apenas um teste, daniel é o meu nome atual cara loko'])
    #print(embedding_matrix)
    # TODO: Start using simple SVMdf = df.groupby('Author').filter(lambda x: len(x) > 1)
    # convert_data('../../data/students_exercises/')
    df = pd.read_csv('../../data/parsed-data/data2.csv')
    result = []
    embbedings = PortugueseTextualProcessing().load_vector_2()
    for text in df['Text']:
        print(text)
        embedding_matrix = PortugueseTextualProcessing().build_embedding_matrix(embbedings, [text])
        print(embedding_matrix)
        result.append(embedding_matrix)

    output = []
    for idx, text in enumerate(result):
        print(idx)
        print(df['Author'][idx])
        output.append([result[idx], df['Author'][idx]])

    #TODO: Convert array into csv file
    print(output)


    # df = remove_single_class_entries(df, 'Author')
    # CSVReader().export_dataframe(df, '../../student_data2.csv')
    #
    # save_converted_stylo_data()
    #
    #
    #
    # df = pd.read_csv('../../data/parsed-data/stylo.csv')
    # df = remove_single_class_entries(df, 'Classe(Autor)')
    # authors = df.pop('Classe(Autor)')
    # df_norm = (df - df.mean()) / (df.max() - df.min())
    # df_norm['Classe(Autor)'] = authors
    #
    # CSVReader().export_dataframe(df_norm, 'stylo2.csv')


    #show_column_distribution('Classe(Autor)')
    #prepare_train_data(df)




