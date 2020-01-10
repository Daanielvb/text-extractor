from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import math
from src.parser.PortugueseTextualProcessing import *
from spellchecker import SpellChecker


class StyloDocument(object):

    DEFAULT_AUTHOR = "Unknown"

    def __init__(self, file_content, author=DEFAULT_AUTHOR):
        self.author = author
        self.file_content = file_content.lower()
        self.tokens = word_tokenize(self.file_content, language='portuguese')
        self.text = Text(self.tokens)
        self.fdist = FreqDist(self.text)
        self.sentences = sent_tokenize(self.file_content, language='portuguese')
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.file_content.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]
        self.punctuation = [".", ",", ";", "-", ":"]
        self.white_spaces = len(self.file_content.split(' '))
        self.tagged_sentences = PortugueseTextualProcessing.postag(self.tokens)
        self.tagfdist = FreqDist([b for [(a, b)] in self.tagged_sentences])
        self.spell = SpellChecker(language='pt')

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

    def max_word_len(self):
        words = set(word_tokenize(self.file_content, language='portuguese'))
        return max([len(word) for word in words])

    def type_token_ratio(self):
        return (len(set(self.text)) / len(self.text)) * 100

    def unique_words_per_hundred(self):
        return self.type_token_ratio() / 100.0 * 100.0 / len(self.text)

    def document_len(self):
        return sum(self.sentence_chars)

    def count_characters_frequency(self, character_list):
        return (len([word for word in self.file_content if word in character_list])) / len(self.text)

    def count_consonant_frequency(self):
        character_list = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y','x', 'z']
        return (len([word for word in self.file_content if word in character_list])) / len(self.text)

    def local_hapax_legommena_frequency(self):
        return (len(self.fdist.hapaxes()))/len(self.text.tokens)

    def collocations_frequency(self, size):
        return (len(self.text.collocation_list(window_size=size)))/len(self.text.tokens)

    def most_frequent_word_size(self):
        return FreqDist(len(w) for w in self.text).max()

    def mean_frequent_word_size(self):
        return FreqDist(len(w) for w in self.text).most_common(3)[1][0]

    def guiraud_R_measure(self):
        return (len(set(self.text)))/math.sqrt(len(self.text))

    def herdan_C_measure(self):
        # log V(N)/log N
        return (math.log2(len(set(self.text))))/math.log2(len(self.text))

    def herdan_V_measure(self):
        # N ^ C
        return math.pow(len(self.text), self.herdan_C_measure())

    def K_measure(self):
        # log V(N)/log(log(N))
        return (math.log2(len(set(self.text)))) / math.log2(math.log2(len(self.text)))

    def dugast_U_measure(self):
        # log ^ 2 N/log(N) - log V(N)
        return (math.pow(math.log2(len(self.text)), 2)) / (math.log2(len(self.text)) - math.log2(len(set(self.text))))

    def maas_A_measure(self):
        #a ^ 2 = logN - logV(N)/log ^ 2 N
        return math.sqrt((math.log2(len(self.text)) - math.log2(len(set(self.text))))
                          / math.pow(math.log2(len(self.text)), 2))

    def LN_measure(self):
        # 1 - V(N) ^ 2/ V(N) ^ 2 log N
        return (1 - math.pow(len(set(self.text)),2)) / (math.pow(len(set(self.text)), 2) * math.log2(len(self.text)))

    def honores_H_measure(self):
        return (len(self.fdist.hapaxes()))/len(set(self.text))

    def spell_miss_check_frequency(self):
        return (len(self.spell.unknown(self.text))) / len(self.text)

    # TODO: global Hapax legomena freq -  might need to have the whole text in a string in order to calculate that.
    # TODO: Number of long words
    @classmethod
    def csv_header(cls):
        return (
            ['DiversidadeLexica', 'TamanhoMedioDasPalavras', 'TamanhoMedioSentencas', 'StdevSentencas', 'TamanhoMedioParagrafos',
             #'TamanhoDocumento',
             'Ponto','Virgulas', 'PontoEVirgula','Exclamacoes', 'DoisPontos', 'Travessao', 'E',
             'Mas', 'Porem', 'Se', 'Isto', 'Mais', 'Precisa', 'Pode', 'Esse', 'Muito', 'FreqAdjetivos', 'FreqAdv',
             'FreqArt', 'FreqSubs', 'FreqPrep', 'FreqVerbos', 'FreqConj', 'FreqPronomes', 'FreqTermosNaoTageados', 'FreqPalavrasErradas',
             'FreqVogais', 'FreqLetrasA', 'FreqLetrasE', 'FreqLetrasI', 'FreqLetrasO', 'FreqLetrasU', 'FrequenciaConsoantes',
             'FrequenciaDeHapaxLegomenaLocal','FrequenciaDeBigrams', 'FrequenciaDeTrigrams', 'FrequenciaDeQuadrigrams',
             'TamanhoMaisFrequenteDePalavras', 'TamanhoMaiorPalavra','GuiraudR', 'HerdanC', 'HerdanV', 'MedidaK',
             'DugastU', 'MaasA', 'MedidaLN', 'HonoresH', 'Classe(Autor)']
        )

    def csv_output(self):
        # 52 {} + class {} (T53)
        return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}," \
               "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'{}'".format(
            round(self.type_token_ratio(), 5),
            round(self.mean_word_len(), 5),
            round(self.mean_sentence_len(), 5),
            round(self.std_sentence_len(), 5),
            round(self.mean_paragraph_len(), 5),
            # self.document_len(),
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
            self.spell_miss_check_frequency(),
            self.count_characters_frequency(['a', 'e', 'i', 'o', 'u']),
            self.count_characters_frequency(['a']),
            self.count_characters_frequency(['e']),
            self.count_characters_frequency(['i']),
            self.count_characters_frequency(['o']),
            self.count_characters_frequency(['u']),
            self.count_consonant_frequency(),
            round(self.local_hapax_legommena_frequency(), 5),
            self.collocations_frequency(2),
            self.collocations_frequency(3),
            self.collocations_frequency(4),
            self.mean_frequent_word_size(),
            self.max_word_len(),
            round(self.guiraud_R_measure(), 5),
            round(self.herdan_C_measure(), 5),
            round(self.herdan_V_measure(), 5),
            round(self.K_measure(), 5),
            round(self.dugast_U_measure(), 5),
            round(self.maas_A_measure(), 5),
            round(self.LN_measure(), 5),
            round(self.honores_H_measure(), 5),
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
        print('Commas                   :', self.term_frequency(','))
        print('Semicolons               :', self.term_frequency(';'))
        print('Quotations               :', self.term_frequency('\"'))
        print('Exclamations             :', self.term_frequency('!'))
        print('Colons                   :', self.term_frequency(':'))
        print('Hyphens                  :', self.term_frequency('-'))  # m-dash or n-dash?
        print('Double Hyphens           :', self.term_frequency('--'))  # m-dash or n-dash?

        print(">>> Analise lexica (per 100 tokens) <<<\n")
        print('e                      :', self.term_frequency('e'))
        print('mas                      :', self.term_frequency('mas'))
        print('porém                  :', self.term_frequency('porém'))
        print('se                       :', self.term_frequency('se'))
        print('isto                     :', self.term_frequency('isto'))
        print('more                     :', self.term_frequency('mais'))
        print('precisa                     :', self.term_frequency('precisa'))
        print('pode                    :', self.term_frequency('pode'))
        print('esse                     :', self.term_frequency('esse'))
        print('muito                     :', self.term_frequency('muito'))
        print("\n")
