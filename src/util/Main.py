from src.parser.PortugueseTextualProcessing import *
from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *
import pandas as pd


def convert_data(base_path):
    clean_txt_content, txt_file_names = FileUtil.convert_files(base_path)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_paths = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    CSVReader.write_files('../../data/parsed-data/', file_paths, 'data2.csv', files_content)


if __name__ == '__main__':
    # TODO: Explore a little bit better pandas
    # TODO: Finish the text-to-features conversion
    # TODO: Start using simple SVM
    convert_data('../../data/students_exercises/')
    data = pd.read_csv('../../data/parsed-data/data2.csv', encoding='utf-8')
    print(data['Author'])
    #results = CSVReader.read_csv('../../data/parsed-data/data.csv')
    #for result in results:
    #   result.text_output()



