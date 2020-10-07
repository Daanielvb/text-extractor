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
        for std in StudentConstant.STOPWORDS:
            file_content = re.sub(r"\b%s\b" % std.lower(), '', file_content)
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
