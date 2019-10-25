import docx2txt
from src.parser.FileUtil import *
from src.Cleaner import *


class DOCReader(FileUtil):

    def __init__(self):
        self.extensions = ["*.doc", "*.docx", "*.odt"]
        pass

    @staticmethod
    def extract_content(file_paths):
        result = []
        for file in file_paths:
            extracted_text = docx2txt.process(file)
            result.append(extracted_text.upper())
        return result, DOCReader().get_file_name(file_paths)

    @staticmethod
    def convert_docs(base_path):
        # TODO: Write the files with their content with a significant name
        contents, file_names = DOCReader().extract_content(DOCReader().get_files_by_extension(base_path, DOCReader().extensions))
        clean_content = Cleaner().remove_headers(contents)
        DOCReader().write_files('../../data/parsed-data/', file_names, clean_content, '.txt')
