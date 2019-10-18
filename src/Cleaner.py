import re


class Cleaner:
    STUDENT_PATTERN = "((Aluno|Aluna) *:?\D*)"
    TEACHER_PATTERN = "((Professor|Professora) *:?\D*)"
    COURSE_PATTERN = "(Disciplina|Materia) *:?\D*"

    def __init__(self):
        pass

    @staticmethod
    def remove_pattern(pattern, text):
        regex = re.compile(pattern, re.IGNORECASE)
        return re.sub(regex, text)


