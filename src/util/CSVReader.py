import csv
from src.parser.StyloDocument import *
from src.util.FileUtil import *


class CSVReader:

    def __init__(self):
        pass

    @staticmethod
    def write_files(folder_path, files, file_name, file_contents):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(folder_path + file_name, 'w', encoding="utf-8") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Text', 'Author'])
            for idx, content in enumerate(file_contents):
                filewriter.writerow([content, FileUtil.convert_author_name(files[idx].lower())])

    @staticmethod
    def write_stylo_features(folder_path, file_name, stylo_objects):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(folder_path + file_name, 'w', encoding="utf-8") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(StyloDocument.csv_header())
            for stylo_obj in stylo_objects:
                row = stylo_obj.csv_output().split(",")
                filewriter.writerow(row)

    @staticmethod
    def read_csv(file, verbose=False):
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
                        results.append(StyloDocument(row[0], row[1]))
                        if verbose:
                            print('Line count:' + str(line_count) + ' Text content:' + row[0])
                        line_count += 1
        return results

    @staticmethod
    def export_dataframe(dataframe, file_name):
        dataframe.to_csv(file_name, index=None, header=True)
