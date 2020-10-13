import csv
from src.parser.StyloDocument import *
from src.util.FileUtil import *


class CSVReader:

    def __init__(self):
        pass

    @staticmethod
    def write_files(folder_path, files, file_name, file_contents, author_naming=True, columns=['Text', 'Author'], folders=[]):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(folder_path + file_name, 'w', encoding="utf-8") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(columns)
            for idx, content in enumerate(file_contents):
                row = []
                row.append(content)
                if not author_naming:
                    author_name = FileUtil.another_author_convention(files[idx])
                    row.append(folders[idx])
                else:
                    author_name = FileUtil.convert_author_name(files[idx])
                row.append(author_name)
                filewriter.writerow(row)

    @staticmethod
    def write_stylo_features(folder_path, file_name, stylo_objects):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(folder_path + file_name, 'w', encoding="utf-8") as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(StyloDocument.csv_header())
            for stylo_obj in stylo_objects:
                row = stylo_obj.csv_output().split(",")
                filewriter.writerow(row)

    @staticmethod
    def transform_text_to_stylo_text(file, verbose=False):
        print('reading the following file:' + file)
        results = []
        with open(file, encoding='utf-8') as f:
            reader = csv.reader(f)
            line_count = 0
            for row in reader:
                if line_count == 0:
                    if verbose:
                        print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if row:
                        # results.append(StyloDocument(file_content=row[0], author=row[1], author=row[2]))
                        results.append(StyloDocument(file_content=row[0], author=row[1]))
                        if verbose:
                            print('Line count:' + str(line_count) + ' Text content:' + row[0])
                        line_count += 1
        return results

    @staticmethod
    def export_dataframe(dataframe, file_name):
        dataframe.to_csv(file_name + '.csv', index=None, header=True)
