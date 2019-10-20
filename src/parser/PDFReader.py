from os import listdir
from os.path import isfile, join
import fnmatch
import os
import PyPDF2
import pdfminer
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io


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
    def extract_pdf_content(files_path):
        result = []
        for file in files_path:
            fp = open(file, 'rb')
            rsrc_mgr = PDFResourceManager()
            retstr = io.StringIO()
            codec = 'utf-8'
            la_params = LAParams()
            device = TextConverter(rsrc_mgr, retstr, codec=codec, laparams=la_params)
            # Create a PDF interpreter object.
            interpreter = PDFPageInterpreter(rsrc_mgr, device)
            # Process each page contained in the document.

            pdf_content = []
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
                data = retstr.getvalue().upper()
                pdf_content.append(data)
            result.append(' '.join(pdf_content))
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





