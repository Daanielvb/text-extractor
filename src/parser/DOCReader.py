from src.parser.FileUtil import *
import docx2txt


class DOCReader:

    def __init__(self):
        self.extensions = ["*.doc", "*.docx", "*.odt"]
        pass

    @staticmethod
    def extract_content(file_paths):
        result = []
        for file in file_paths:
            extracted_text = docx2txt.process(file)
            result.append(extracted_text.upper())
        return result

    def get_files(self, folder_path):
        return FileUtil().get_files_by_extension(folder_path, self.extensions)
