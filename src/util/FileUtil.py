import os
import functools
import itertools
import fnmatch
from pickle import dump
import pickle
from src.Cleaner import *


class FileUtil:

    def __init__(self):
        self.extensions = ["*.txt"]
        pass

    @staticmethod
    def get_words_from_file(file_path):
        file = open(file_path, 'r', encoding='iso-8859-1')
        return [line.split('\n')[0].lower() for line in file.readlines()]

    @staticmethod
    def get_files_by_extension(folder_path, extensions):
        result = []
        for f in FileUtil().find_files(folder_path, extensions):
            result.append(f)
        return result

    @staticmethod
    def find_files(dir_path: str = None, patterns: [str] = None) -> [str]:
        path = dir_path or "."
        path_patterns = patterns or ["*"]

        for root_dir, dir_names, file_names in os.walk(path):
            filter_partial = functools.partial(fnmatch.filter, file_names)

            for file_name in itertools.chain(*map(filter_partial, path_patterns)):
                yield os.path.join(root_dir, file_name)

    @staticmethod
    def get_file_name(file_paths):
        return [os.path.splitext(os.path.basename(os.path.normpath(path)))[0] for path in file_paths]

    @staticmethod
    def convert_files(base_path):
        files = FileUtil().get_files_by_extension(base_path, FileUtil().extensions)
        content = Cleaner().remove_patterns(FileUtil.extract_file_content(files))
        return content, FileUtil.get_file_name(files)

    @staticmethod
    def extract_file_content(file_paths):
        result = []
        for path in file_paths:
            with open(path, 'r', errors='ignore') as file:
                result.append(file.read().lower())
        return result

    @staticmethod
    def write_pickle_file(file_name, content):
        output = open(file_name, 'wb')
        dump(content, output, -1)
        output.close()

    @staticmethod
    def merge_contents(*args):
        result = []
        for arg in args:
            result.extend(arg)
        return result

    @staticmethod
    def convert_author_name(name):
        if ';' not in name:
            return ''.join([name[0] for name in name.split(" ")[::-1]])
        else:
            return ''.join([name[0:3] for name in name.split(";")])

    @staticmethod
    def load_ner_pickle(filename='cat-min_lldelta_0.pickle'):
        with open(filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            tagger = u.load()
        return tagger
