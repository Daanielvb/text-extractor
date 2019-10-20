from src.parser.PDFReader import *
from src.Cleaner import *

if __name__ == '__main__':
    files_list = PDFReader().get_files('../../data/pdf')
    pdf_contents = PDFReader().extract_pdf_content(files_list)
    pdf_clean_content = Cleaner().remove_headers(pdf_contents)
    for text in pdf_clean_content:
        print(text)
