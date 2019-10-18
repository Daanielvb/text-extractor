from src.parser.PDFReader import *

if __name__ == '__main__':
    files_list = PDFReader().get_files('../../data/pdf')
    pdf_contents = PDFReader().read_pdf_content(files_list)
    PDFReader().write_file('../../data/converted', files_list, pdf_contents, '.txt')