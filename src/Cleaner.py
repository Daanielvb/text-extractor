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
    INSTITUTION_PATTERN = "(CONGREGAÇÃO DE SANTA DOROTÉIA DO BRASIL|FACULDADE FRASSINETTI DO RECIFE|FAFIRE)\n*"
    DATE_PATTERN = "RECIFE[, ]?\d*"

    def __init__(self):
        pass

    @staticmethod
    def remove_student_names(file_content):
        result = []
        for std in StudentConstant.STOPWORDS:
            file_content = re.sub(r"\b%s\b" % std.upper(), '', file_content)
        result.append(file_content)
        return result

    @staticmethod
    def remove_headers(text_list):
        result = []
        for text in text_list:
            # text = re.sub(Cleaner.STUDENT_PATTERN, '', text, flags=re.IGNORECASE)
            # text = re.sub(Cleaner.TEACHER_PATTERN,  '', text, flags=re.IGNORECASE)
            # text = re.sub(Cleaner.COURSE_PATTERN, '', text, flags=re.IGNORECASE)
            text = Cleaner().remove_student_names(text)[0]
            text = text.replace('"', "'")
            text = re.sub(Cleaner.INSTITUTION_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.PERIOD_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.DATE_PATTERN, '', text, flags=re.IGNORECASE)
            # TODO: Verify if results are better using PortugueseTextualProcessing().remove_stopwords()
            result.append(Cleaner().format(text))
        return result

    @staticmethod
    def format(text):
        #text = re.sub('\n', ' ', text)
        regex = r"(\n{2,} )"
        text = re.sub(regex, '\n', text)
        text = re.sub('\n{3,}', '\n\n', text)
        text = re.sub(r"[\t]*", "", text)
        return text.strip()
