from os import listdir
from os.path import isfile, join
import fnmatch
import os
import PyPDF2


class PDFReader:

    def __init__(self):
        pass

    @staticmethod
    def read_pdf_content(file_paths):
        result = []
        for path in file_paths:
            pdf_file_object = open(path, 'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file_object)
            content = ''
            for page_idx in range(pdf_reader.getNumPages()):
                content += pdf_reader.getPage(page_idx).extractText()
            result.append(content)
        return result

    @staticmethod
    def write_file(folder, file_name, file_contents, extension):
        for file, content in file_contents, file_name:
            with open(file + extension, 'w') as f:
                f.write(content)

    @staticmethod
    def get_files_for_extraction(folder_path):
        return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    @staticmethod
    def get_files(folder_path):
        result = []
        for path, dirs, files in os.walk(folder_path):
            for f in fnmatch.filter(files, '*.pdf'):
                fullname = os.path.abspath(os.path.join(path, f))
                result.append(fullname)
        return result





