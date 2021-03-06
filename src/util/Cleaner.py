import re
from src.util.StudentConstant import *


class Cleaner:

    STUDENT_PATTERN = "((Aluno|Aluna|Alunos|Alunas|Grupo|Equipe|Autor|Autora|Autores|Discente) *:+)"
    # TODO: Consider adding a ; instead of just : after the student name
    # TODO: This regex is cleaning all the content for some texts, like A6/julia marley
    # ("((Aluno|Aluna|Alunos|Alunos|Grupo|Equipe|Aluna|Autor|Autora|Discente) *:+\D*)\n*"
    TEACHER_PATTERN = "((Professor|Professora|Docente) *:+\D*)\n*"
    COURSE_PATTERN = "(Disciplina|Materia|Cadeira|Curso) *:+\D*\n*"
    PERIOD_PATTERN = "(Periodo|Período|Ano|Ano/Semestre|Semestre) *:+\w*\n*"
    INSTITUTION_PATTERN = "(congregação de santa dorotéia do brasil|faculdade frassinetti do recife|fafire)\n*"
    DATE_PATTERN = "recife[, ]?\d*"
    URL_PATTERN = r"http\S+"
    OPEN_PARENTHESIS = r"\("
    CLOSING_PARENTHESIS = r"\)"

    def __init__(self):
        pass

    @staticmethod
    def remove_student_names(file_content):
        result = []
        for stopword in StudentConstant.STOPWORDS:
            file_content = re.sub(r"\b%s\b" % stopword, '', file_content, flags=re.IGNORECASE)
        result.append(file_content)
        return result

    @staticmethod
    def remove_patterns(text_list, raw=False):
        result = []
        for text in text_list:
            # text = re.sub(Cleaner.STUDENT_PATTERN, '', text, flags=re.IGNORECASE)
            # text = re.sub(Cleaner.TEACHER_PATTERN,  '', text, flags=re.IGNORECASE)
            # text = re.sub(Cleaner.COURSE_PATTERN, '', text, flags=re.IGNORECASE)
            text = Cleaner().remove_student_names(text)[0]
            text = text.replace('"', "'")
            text = re.sub(Cleaner.OPEN_PARENTHESIS, "( ", text)
            text = re.sub(Cleaner.CLOSING_PARENTHESIS, " )", text)
            text = re.sub(Cleaner.INSTITUTION_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.PERIOD_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.DATE_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.URL_PATTERN, '', text, flags=re.IGNORECASE)
            # TODO: Verify if results are better using PortugueseTextualProcessing().remove_stopwords()
            if not raw:
                text = Cleaner().remove_new_lines(text)
            result.append(text)
        return result

    @staticmethod
    def remove_new_lines(text):
        #text = re.sub('\n', ' ', text)
        regex = r"(\n{2,} )"
        text = re.sub(regex, '\n', text)
        text = re.sub('\n{3,}', '\n\n', text)
        text = re.sub(r"[\t]*", "", text)
        return text.strip()

    @staticmethod
    def remove_varela_authors(df):
        regex_baleia = 'baleia - \d{1,2}\/\d{1,2}\/\d{2,4}'
        regex_ana = 'ana cristina cavalcante\s*\d{1,2} \w{1,4} \d{2,4} - \S{3,5}(min)*'
        regex_adriano = 'adriano gambarini - \d{1,2}\/\d{1,2}\/\d{2,4}'
        regex_ivolnildo = 'ivon(i)*l(d)*o lavôr(\s)*\d{1,2} \w{1,4} \d{2,4} - \S{4,5}(min)*'
        regex_mario = 'mário pinto(\s)*\d{1,2} \w{1,4} \d{2,4} - \S{4,5}(min)*'
        regex_julio = "mais sobr julio preuss - \d{1,2}\/\d{1,2}\/\d{2,4}(julio preussescreveu o livro 'fotografia digital: da compra da câmera à impressão das fotos)*"
        regex_roberto = 'roberto linsker - \d{1,2}\/\d{1,2}\/\d{2,4}'
        df['Text'] = df['Text'].str.replace(regex_baleia, ' ')
        df['Text'] = df['Text'].str.replace(regex_ana, ' ')
        df['Text'] = df['Text'].str.replace(regex_adriano, ' ')
        df['Text'] = df['Text'].str.replace(regex_ivolnildo, ' ')
        df['Text'] = df['Text'].str.replace(regex_mario, ' ')
        df['Text'] = df['Text'].str.replace(regex_julio, ' ')
        df['Text'] = df['Text'].str.replace(regex_roberto, ' ')
        return df
