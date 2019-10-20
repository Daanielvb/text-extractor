import os
import functools
import itertools
import fnmatch


class FileUtil:

    def __init__(self):
        pass

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
    def write_files(folder, files, file_contents, extension):
        for file, content in file_contents, files:
            with open(folder + file + extension, 'w') as f:
                f.write(content)