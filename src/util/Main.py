from src.parser.PortugueseTextualProcessing import *
from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def convert_data(base_path):
    clean_txt_content, txt_file_names = FileUtil.convert_files(base_path)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_paths = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    CSVReader.write_files('../../data/parsed-data/', file_paths, 'data2.csv', files_content)


def prepare_train_data(dataframe):
    X = dataframe['Text']
    y = dataframe['Author']
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    sss.get_n_splits(X, y)
    for train_index, test_index in sss.split(X, y):
        # TODO: Check some NaNs
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def remove_group_works(dataframe):
    dataframe['Author'] = dataframe['Author'].astype('str')
    mask = (dataframe['Author'].str.len() <= 5)
    return dataframe.loc[mask]


def remove_single_class_entries(dataframe):
    #TODO: Check some NaNs
    return dataframe.groupby('Author').filter(lambda x: len(x) > 1)


if __name__ == '__main__':
    # TODO: Finish the text-to-features conversion
    # TODO: Start using simple SVMdf = df.groupby('Author').filter(lambda x: len(x) > 1)
    # convert_data('../../data/students_exercises/')
    # df = pd.read_csv('../../data/parsed-data/data2.csv', encoding='utf-8')
    # df = remove_group_works(df)
    # df = remove_single_class_entries(df)
    # prepare_train_data(df)

    stylo_objs = CSVReader.read_csv('../../data/parsed-data/data2.csv')
    print(stylo_objs[0].csv_output())
    CSVReader.write_stylo_features('../../data/parsed-data/', 'stylo.csv', stylo_objs)



