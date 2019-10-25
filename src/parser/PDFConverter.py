from src.parser.PDFReader import *
from src.parser.DOCReader import *


def convert_data(base_path):
    DOCReader().convert_docs(base_path)
    PDFReader().convert_pdfs(base_path)


if __name__ == '__main__':
    convert_data('../../data/students_exercises/')


