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
    def get_file_name(file_paths):
        return [os.path.splitext(os.path.basename(os.path.normpath(path)))[0] for path in file_paths]

    @staticmethod
    def write_files(folder_path, files, file_contents, extension):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for idx, content in enumerate(file_contents):
            counter = 0
            file_name = folder_path + files[idx] + extension
            #TODO: Check why this approach is not working, might be stuck in a loop
            while not os.path.isfile(file_name) and counter < 10:
                counter += 1
                file_name = folder_path + files[idx] + str(counter) + extension
                with open(file_name, 'w') as f:
                    f.write(content[0])

