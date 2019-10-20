import re


class Cleaner:

    STUDENT_PATTERN = "((Aluno|Aluna|Autor|Autora|Discente) *:+\D*)\n*"
    TEACHER_PATTERN = "((Professor|Professora|Docente) *:+\D*)\n*"
    COURSE_PATTERN = "(Disciplina|Materia|Cadeira|Curso) *:+\D*\n*"
    PERIOD_PATTERN = "(Periodo|Período|Ano|Ano/Semestre|Semestre) *:+\w*\n*"
    INSTITUTION_PATTERN = "(CONGREGAÇÃO DE SANTA DOROTÉIA DO BRASIL|FACULDADE FRASSINETTI DO RECIFE|FAFIRE)\n*"
    DATE_PATTERN = "RECIFE[, ]?\d*"
    SPECIAL_PATTERNS = ["Autor Rodrigo Leandro de Lira dos Santos"]

    def __init__(self):
        pass

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
            result.append(re.sub(Cleaner.DATE_PATTERN, '', text, flags=re.IGNORECASE))
        return result


