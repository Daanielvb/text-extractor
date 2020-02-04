import docx2txt
from src.util.FileUtil import *
from src.Cleaner import *


class DOCReader(FileUtil):

    def __init__(self):
        self.extensions = ["*.doc", "*.docx", "*.odt"]
        pass

    @staticmethod
    def extract_content(file_paths):
        result = []
        for file in file_paths:
            print('extracting content from file:' + file)
            extracted_text = docx2txt.process(file)
            result.append(extracted_text.upper())
        return result

    @staticmethod
    def convert_docs(base_path):
        file_names = DOCReader().get_files_by_extension(base_path, DOCReader().extensions)
        content = Cleaner().remove_headers(DOCReader().extract_content(file_names))
        return content, DOCReader().get_file_name(file_names)
