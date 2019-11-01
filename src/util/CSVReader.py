import csv
import os

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
                filewriter.writerow([content[0], FileUtil.convert_author_name(files[idx].upper())])

    @staticmethod
    def read_csv(file):
        results = []
        with open(file, encoding='utf-8') as f:
            reader = csv.reader(f)
            line_count = 0
            for row in reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    if row:
                        results.append(StyloDocument(row[0],row[1]))
                        print('Line count:' + str(line_count) + ' Text content:' + row[0])
                        line_count += 1
        return results
