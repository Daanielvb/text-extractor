from src.parser.PDFReader import *
from src.Cleaner import *
from src.parser.DOCReader import *


def convert_pdfs(base_path):
    files_list = PDFReader().get_files(base_path)
    pdf_contents = PDFReader().extract_content(files_list)
    pdf_clean_content = Cleaner().remove_headers(pdf_contents)
    # TODO: Write the files with their content with a significant name
    return pdf_clean_content


def convert_docs(base_path):
    files_list = DOCReader().get_files(base_path)
    doc_contents = DOCReader().extract_content(files_list)
    doc_clean_content = Cleaner().remove_headers(doc_contents)
    # TODO: Write the files with their content with a significant name
    return doc_clean_content


if __name__ == '__main__':
    pdfs = convert_pdfs('../../data/pdf')
    docs = convert_docs('../../data/doc')


