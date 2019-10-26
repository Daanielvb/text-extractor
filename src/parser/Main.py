from src.parser.PDFReader import *
from src.parser.DOCReader import *


def convert_data(base_path):
    clean_txt_content, txt_file_names = FileUtil.convert_files(base_path)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_names = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    FileUtil.write_files('../../data/parsed-data/', file_names, files_content)


if __name__ == '__main__':
    convert_data('../../data/students_exercises/')


