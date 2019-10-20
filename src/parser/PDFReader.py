from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from src.parser.FileUtil import *
import io


class PDFReader:

    def __init__(self):
        self.extensions = ["*.pdf"]
        pass

    @staticmethod
    def extract_content(files_path):
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

    def get_files(self, folder_path):
        return FileUtil().get_files_by_extension(folder_path, self.extensions)





