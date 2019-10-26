import re
from src.parser.StudentConstant import *

class Cleaner:

    STUDENT_PATTERN = "((Aluno|Aluna|Alunos|Alunas|Grupo|Equipe|Autor|Autora|Autores|Discente) *:+\D*)\n*"
    # TODO: Consider adding a ; instead of just : after the student name
    # TODO: This regex is cleaning all the content for some texts, like A6/julia marley
    # ("((Aluno|Aluna|Alunos|Alunos|Grupo|Equipe|Aluna|Autor|Autora|Discente) *:+\D*)\n*"
    TEACHER_PATTERN = "((Professor|Professora|Docente) *:+\D*)\n*"
    COURSE_PATTERN = "(Disciplina|Materia|Cadeira|Curso) *:+\D*\n*"
    PERIOD_PATTERN = "(Periodo|Período|Ano|Ano/Semestre|Semestre) *:+\w*\n*"
    INSTITUTION_PATTERN = "(CONGREGAÇÃO DE SANTA DOROTÉIA DO BRASIL|FACULDADE FRASSINETTI DO RECIFE|FAFIRE)\n*"
    DATE_PATTERN = "RECIFE[, ]?\d*"
    SPECIAL_PATTERNS = ["Autor Rodrigo Leandro de Lira dos Santos"]

    def __init__(self):
        pass

    @staticmethod
    def remove_student_names(file_content):
        result = []
        for std in StudentConstant.NAMES:
            file_content = file_content.replace(std.upper(), '')
        result.append(file_content)
        return result

    @staticmethod
    def remove_headers(text_list):
        result = []
        for text in text_list:
            text = re.sub(Cleaner.STUDENT_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.TEACHER_PATTERN,  '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.COURSE_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.INSTITUTION_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.PERIOD_PATTERN, '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.SPECIAL_PATTERNS[0], '', text, flags=re.IGNORECASE)
            text = re.sub(Cleaner.DATE_PATTERN, '', text, flags=re.IGNORECASE)
            text = Cleaner().format(text)
            result.append(Cleaner().remove_student_names(text))
        return result

    @staticmethod
    def format(text):
        text = re.sub('\n', ' ', text)
        text = re.sub('\n{2,}', '', text)
        return text.strip()
