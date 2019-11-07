from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


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

def remove_single_class_entries(dataframe):
    #TODO: Check some NaNs
    return dataframe.groupby('Autor').filter(lambda x: len(x) > 1)


if __name__ == '__main__':
    #df = pd.read_csv('../../data/parsed-data/stylo.csv')
    #df = remove_single_class_entries(df)
    # TODO: Finish the text-to-features conversion
    # TODO: Start using simple SVMdf = df.groupby('Author').filter(lambda x: len(x) > 1)
    #convert_data('../../data/students_exercises/')
    #df = pd.read_csv('../../data/parsed-data/data2.csv', encoding='utf-8')
    #stylo_objs = CSVReader.read_csv('../../data/parsed-data/data2.csv')
    #CSVReader().write_stylo_features('../../data/parsed-data/', 'stylo.csv', stylo_objs)
    df = pd.read_csv('../../data/parsed-data/stylo.csv')
    df = remove_single_class_entries(df)
    prepare_train_data(df)




