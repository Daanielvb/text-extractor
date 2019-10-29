from src.parser.PortugueseTextualProcessing import *
from src.util.PDFReader import *
from src.util.DOCReader import *
from src.util.CSVReader import *


def convert_data(base_path):
    clean_txt_content, txt_file_names = FileUtil.convert_files(base_path)
    clean_doc_content, doc_file_names = DOCReader().convert_docs(base_path)
    clean_pdf_content, pdf_file_names = PDFReader().convert_pdfs(base_path)

    files_content = FileUtil.merge_contents(clean_txt_content, clean_doc_content, clean_pdf_content)
    file_names = FileUtil.merge_contents(txt_file_names, doc_file_names, pdf_file_names)
    CSVReader.write_files('../../data/parsed-data/', file_names, files_content)


if __name__ == '__main__':
    text = 'INTERDISCIPLINARIDADE É QUANDO OCORRE UMA “LIGAÇÃO” ENTRE DUAS OU MAIS DISCIPLINAS, É IMPORTANTE LEMBRAR QUE SOMO NÓS QUE FRAGMENTAMOS AS CIÊNCIAS SEJAM ELAS EXATAS, NATURAIS OU HUMANAS; POR ISSO O TERMO “LIGAÇÃO”. UM EXEMPLO BOM EXEMPLO DE INTERDISCIPLINARIDADE É QUANDO EM UM PROBLEMA OU SITUAÇÃO É PRECISO USAR AS CIÊNCIAS EXATAS JUNTAS DAS NATURAIS: EM UMA SITUAÇÃO EM QUE UM MERGULHADOR PRECISA DESCER A UMA DETERMINADA PROFUNDIDADE PARA ENCONTRAR UM ORGANISMO MARINHO; NESSE CASO É PRECISO CONSIDERAR A PRESSÃO DA ÁGUA (FÍSICA), A PROFUNDIDADE EM RELAÇÃO A PRESSÃO (MATEMÁTICA), O TEMPO DE ADAPTAÇÃO DO CORPO HUMANO (BIOLOGIA), ALÉM DE OUTROS FATORES. '
    print (PortugueseTextualProcessing().postag(PortugueseTextualProcessing().tokenize(text)))

    #TODO: Check https://github.com/fmaruki/Nltk-Tagger-Portuguese answers to see if the picle can be loaded later
    #results = CSVReader.read_csv('../../data/parsed-data/data.csv')
    #for result in results:
        #result.text_output()
    #convert_data('../../data/students_exercises/')


