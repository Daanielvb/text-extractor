from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from src.util.FileUtil import *
from src.util.Cleaner import *
import io


class PDFReader(FileUtil):

    def __init__(self):
        self.extensions = ["*.pdf"]
        pass

    @staticmethod
    def extract_content(files_path, raw=False):
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
                if raw:
                    data = retstr.getvalue()
                else:
                    data = retstr.getvalue().lower()
                pdf_content.append(data)
            result.append(' '.join(pdf_content))
        return result

    @staticmethod
    def convert_pdfs(base_path, raw):
        file_names = PDFReader().get_files_by_extension(base_path, PDFReader().extensions)
        content = Cleaner().remove_patterns(PDFReader().extract_content(file_names, raw), raw)
        return content, PDFReader().get_file_name(file_names)






