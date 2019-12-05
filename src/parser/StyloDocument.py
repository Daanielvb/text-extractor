DEFAULT_AUTHOR = "Unknown"
from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import numpy as np
from src.parser.PortugueseTextualProcessing import *


class StyloDocument(object):

    def __init__(self, file_content, author=DEFAULT_AUTHOR):
        self.author = author
        self.file_content = file_content.lower()
        self.tokens = word_tokenize(self.file_content, language='portuguese')
        self.text = Text(self.tokens)
        self.fdist = FreqDist(self.text)
        self.sentences = sent_tokenize(self.file_content, language='portuguese')
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        # TODO: Fix paragraphs, all records are being set to 1, might be related to \n replacing at data extraction
        self.paragraphs = [p for p in self.file_content.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]
        self.collocations = self.text.collocation_list()
        self.punctuation = [".", ",", ";", "-", ":"]
        self.white_spaces = len(self.file_content.split(' '))
        self.tagged_sentences = PortugueseTextualProcessing.postag(self.tokens)
        self.tagfdist = FreqDist([b for [(a, b)] in self.tagged_sentences])

    def get_class_frequency_by_start(self, tag_start):
        count = 0
        for tag in self.tagfdist.keys():
            if tag.startswith(tag_start):
                count += self.tagfdist[tag]
        return count/self.tagfdist.N()

    def tag_frequency(self, tag):
        return self.tagfdist.freq(tag)

    def term_per_thousand(self, term):
        """
        term       X
        -----  = ------
          N       1000
        """
        return (self.fdist[term] * 1000) / self.fdist.N()

    def term_per_hundred(self, term):
        """
        term       X
        -----  = ------
          N       100
        """
        return (self.fdist[term] * 100) / self.fdist.N()

    def mean_sentence_len(self):
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        return np.mean(self.paragraph_word_length)

    def std_paragraph_len(self):
        return np.std(self.paragraph_word_length)

    def vocabulary(self):
        return [v for v in sorted(set(self.sentences)) if v not in self.punctuation]

    def mean_word_len(self):
        words = set(word_tokenize(self.file_content, language='portuguese'))
        word_chars = [len(word) for word in words]
        return sum(word_chars) / float(len(word_chars))

    def type_token_ratio(self):
        return (len(set(self.text)) / len(self.text)) * 100

    def unique_words_per_hundred(self):
        return self.type_token_ratio() / 100.0 * 100.0 / len(self.text)

    def document_len(self):
        return sum(self.sentence_chars)

    def count_characters(self, character_list):
        return len([word for word in self.file_content if word in character_list])

    #TODO: global Hapax legomena freq -  might need to have the whole text in a string in order to calculate that.
    def local_hapax_legommena_frequency(self):
        return (len(self.fdist.hapaxes()))/len(self.text.tokens)

    def collocations_frequency(self):
        return (len(self.collocations))/len(self.text.tokens)

    @classmethod
    def csv_header(cls):
        return (
            ['DiversidadeLexica', 'TamanhoMedioDasPalavras', 'TamanhoMedioSentencas', 'StdevSentencas', 'TamanhoMedioParagrafos','TamanhoDocumento',
             'Ponto','Virgulas', 'PontoEVirgula','Exclamacoes', 'DoisPontos', 'Travessao', 'E',
             'Mas', 'Porem', 'Se', 'Isto', 'Mais', 'Precisa', 'Pode', 'Esse', 'Muito', 'FreqAdjetivos', 'FreqAdv',
             'FreqArt', 'FreqSubs', 'FreqPrep', 'FreqVerbos', 'FreqConj', 'FreqPronomes', 'TermosNaoTageados',
             'Vogais', 'LetrasA', 'LetrasE', 'LetrasI', 'LetrasO', 'LetrasU', 'FrequenciaDeHapaxLegomenaLocal',
             'CollocationsFrequency', 'Classe(Autor)']
        )

    def csv_output(self):
        # 39 {} + class {}
        return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'{}'".format(
            self.type_token_ratio(),
            self.mean_word_len(),
            self.mean_sentence_len(),
            self.std_sentence_len(),
            self.mean_paragraph_len(),
            self.document_len(),
            self.term_per_hundred('.'),
            self.term_per_hundred(','),
            self.term_per_hundred(';'),
            self.term_per_hundred('!'),
            self.term_per_hundred(':'),
            self.term_per_hundred('-'),
            self.term_per_hundred('e'),
            self.term_per_hundred('mas'),
            self.term_per_hundred('porém'),
            self.term_per_hundred('se'),
            self.term_per_hundred('isto'),
            self.term_per_hundred('mais'),
            self.term_per_hundred('precisa'),
            self.term_per_hundred('pode'),
            self.term_per_hundred('esse'),
            self.term_per_hundred('muito'),
            self.tag_frequency('adj'),
            self.tag_frequency('adv'),
            self.tag_frequency('art'),
            self.tag_frequency('n'),
            self.tag_frequency('prp'),
            self.get_class_frequency_by_start('v'),
            self.get_class_frequency_by_start('conj'),
            self.get_class_frequency_by_start('pron'),
            self.tag_frequency('notfound'),
            self.count_characters(['a', 'e', 'i', 'o', 'u']),
            self.count_characters(['a']),
            self.count_characters(['e']),
            self.count_characters(['i']),
            self.count_characters(['o']),
            self.count_characters(['u']),
            self.local_hapax_legommena_frequency(),
            self.collocations_frequency(),
            self.author,
        )

    def text_output(self):
        print("##############################################")

        print("Author: ", self.author)
        print("\n")

        print(">>> Phraseology Analysis <<<\n")

        print("Lexical diversity        :", self.type_token_ratio())
        print("Mean Word Length         :", self.mean_word_len())
        print("Mean Sentence Length     :", self.mean_sentence_len())
        print("STDEV Sentence Length    :", self.std_sentence_len())
        print("Mean paragraph Length    :", self.mean_paragraph_len())
        print("Document Length          :", self.document_len())
        print("\n")

        print(">>> Analise de pontuacoes (per 100 tokens) <<<\n")
        print('Commas                   :', self.term_per_hundred(','))
        print('Semicolons               :', self.term_per_hundred(';'))
        print('Quotations               :', self.term_per_hundred('\"'))
        print('Exclamations             :', self.term_per_hundred('!'))
        print('Colons                   :', self.term_per_hundred(':'))
        print('Hyphens                  :', self.term_per_hundred('-'))  # m-dash or n-dash?
        print('Double Hyphens           :', self.term_per_hundred('--'))  # m-dash or n-dash?

        print(">>> Analise lexica (per 100 tokens) <<<\n")
        print('e                      :', self.term_per_hundred('e'))
        print('mas                      :', self.term_per_hundred('mas'))
        print('porém                  :', self.term_per_hundred('porém'))
        print('se                       :', self.term_per_hundred('se'))
        print('isto                     :', self.term_per_hundred('isto'))
        print('more                     :', self.term_per_hundred('mais'))
        print('precisa                     :', self.term_per_hundred('precisa'))
        print('pode                    :', self.term_per_hundred('pode'))
        print('esse                     :', self.term_per_hundred('esse'))
        print('muito                     :', self.term_per_hundred('muito'))
        print("\n")
