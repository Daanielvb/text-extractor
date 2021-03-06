from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import math
from src.parser.PortugueseTextualProcessing import *
from spellchecker import SpellChecker


class StyloDocument(object):

    DEFAULT_AUTHOR = "Unknown"

    def __init__(self, file_content, author=DEFAULT_AUTHOR):
        self.author = author.strip()
        self.raw_content = file_content
        self.file_content = file_content.lower()
        self.tokens = PortugueseTextualProcessing.tokenize(self.file_content)
        self.text = Text(self.tokens)
        self.fdist = FreqDist(self.text)
        self.sentences = sent_tokenize(self.file_content, language='portuguese')
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.file_content.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]
        self.punctuation = [".", ",", ";", "-", ":"]
        self.ner_entities = ['ABSTRACCAO', 'ACONTECIMENTO', 'COISA', 'LOCAL',
                             'ORGANIZACAO', 'OBRA', 'OUTRO', 'PESSOA', 'TEMPO', 'VALOR']
        self.white_spaces = len(self.file_content.split(' '))

        self.rich_tags = RichTags(PortugueseTextualProcessing.get_rich_tags(self.file_content), len(self.text))
        self.tagged_sentences = PortugueseTextualProcessing.postag(self.tokens)
        self.tagfdist = FreqDist([b for [(a, b)] in self.tagged_sentences])
        self.ner_tags = PortugueseTextualProcessing.ner_chunks(self.tokens)
        self.ner_ftags = FreqDist(self.ner_tags)
        self.spell = SpellChecker(language='pt')
        self.ROUNDING_FACTOR = 4
        self.LINE_BREAKS = ['\n', '\t', '\r']

    def get_tag_count_by_start(self, tag_start):
        count = 0
        for tag in self.tagfdist.keys():
            if tag.startswith(tag_start):
                count += self.tagfdist[tag]
        return count

    def get_class_frequency_by_start(self, tag_start):
        return self.get_tag_count_by_start(tag_start)/self.tagfdist.N()

    def get_total_not_found(self):
        """"The wn is not being reliable so far"""
        nf_tokens = self.get_tokens_by_tag('notfound')
        return len([i for i in nf_tokens if len(wn.synsets(i, lang='por')) == 0])

    def tag_frequency(self, tag):
        return self.tagfdist.freq(tag)

    def entity_frequency(self, tag):
        return self.ner_ftags.freq(tag)

    def get_tokens_by_tag(self, tag):
        return [i[0][0] for i in self.tagged_sentences if i[0][1] == tag]

    def get_long_sentence_freq(self):
        return (len([i for i in self.sentence_word_length if i < PortugueseTextualProcessing.LONG_SENTENCE_SIZE]))/len(self.sentences)

    def get_short_sentence_freq(self):
        return (len([i for i in self.sentence_word_length if i < PortugueseTextualProcessing.SHORT_SENTENCE_SIZE]))/len(self.sentences)

    def get_long_short_sentence_ratio(self):
        """"RF FOR PAN 15"""
        return len([i for i in self.sentence_word_length if i < PortugueseTextualProcessing.LONG_SENTENCE_SIZE])/(len([i for i in self.sentence_word_length if i < PortugueseTextualProcessing.SHORT_SENTENCE_SIZE]))

    def get_sentence_starting_tags_ratio(self, tag):
        count = [i[0][1] for i in self.tagged_sentences].count(tag)
        return count/len(self.sentences)

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

    def flesh_index(self):
        idx, value = PortugueseTextualProcessing().get_ptBR_flesch_index(self.tokens, self.get_phrases())
        return idx

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

    def get_phrases(self):
        return [i for i in self.file_content.split('.') if i != '']

    def mean_syllables_per_word(self):
        _, syllable_count = PortugueseTextualProcessing().get_syllable_counts(self.tokens)
        return syllable_count/len(self.tokens)

    def characters_frequency(self, character_list):
        return self.frequency([word for word in self.file_content if word in character_list])

    def digits_frequency(self):
        return self.frequency([word for word in self.file_content if word.isdigit()])

    def line_breaks_frequency(self):
        return self.frequency([word for word in self.file_content if word in self.LINE_BREAKS])

    def count_consonant_frequency(self):
        character_list = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w',
                          'y', 'x', 'z']
        return self.frequency([word for word in self.file_content if word in character_list])

    def camel_case_frequency(self):
        return self.frequency([word for word in self.raw_content.split(' ') if word and word[0].isupper() and (len(word) == 1 or word[1].islower())])

    def local_hapax_legommena_frequency(self):
        return (len(self.fdist.hapaxes()))/len(self.text.tokens)

    def collocations_frequency(self, size):
        """words that often appear consecutively in the window_size"""
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
        return self.frequency(self.spell.unknown(self.text))

    def noun_phrases(self):
        return PortugueseTextualProcessing().get_number_of_noun_phrases(self.tokens) / len(self.text)

    def verb_phrases(self):
        return self.frequency(PortugueseTextualProcessing().get_number_of_verb_phrases(self.file_content))

    def monosyllables(self):
        return PortugueseTextualProcessing().get_monosyllable_counts(self.tokens) / len(self.text)

    def repeated_words_frequency(self):
        repeated_words = list(filter(lambda x: x[1] >= 2, FreqDist(PortugueseTextualProcessing().remove_stopwords(self.tokens)).items()))
        return self.frequency(repeated_words)

    def stop_word_freq(self):
        clean_words = PortugueseTextualProcessing().remove_stopwords(self.tokens)
        return (len(self.tokens) - len(clean_words)) / len(self.text)

    def get_logical_operator_frequency(self):
        return self.frequency([token for token in self.tokens if token in PortugueseTextualProcessing.LOGICAL_OPERATORS])

    def get_tags_freq(self, tags):
        count = 0
        for tag in tags:
            count += self.get_tag_count_by_start(tag)
        return count/len(self.tokens)

    def find_quotes(self):
        """Improve this method to retrieve quotes based on Patterns and special words
        egs: p.43;  segundo (autor, ano)
        """
        return self.characters_frequency(['“', '”'])

    def frequency(self, input_values):
        return len(input_values) / len(self.text)

    @classmethod
    def csv_header(cls):
        return (
            ['DiversidadeLexica', 'TamanhoMedioDasPalavras', 'TamanhoMedioSentencas', 'StdevSentencas', 'TamanhoMedioParagrafos',
             'StdevTamParagrafos', 'FrequenciaDeParagrafos','FrequenciaPalavrasDuplicadas', 'MediaSilabasPorPalavra',

             'Monossilabas',

             'Ponto','Virgulas', 'Exclamacoes', 'DoisPontos', 'Citacoes', 'QuebrasDeLinha', 'Digitos',

             'Adjetivos', 'Adverbios','Artigos', 'Substantivos', 'Preposicoes', 'Verbos','VerbosPtcp', 'Conjuncoes',
             'Pronomes', 'PronomesPorPreposicao','TermosNaoTageados', 'PalavrasDeConteudo', 'PalavrasFuncionais',
             'FrasesNominais', 'FrasesVerbais', 'GenMasc', 'GenFem', 'SemGenero', 'Singular', 'Plural',

             'PrimeiraPessoa', 'TerceiraPessoa','Passado','Presente','Futuro',

             'TotalEntidadesNomeadas', 'EntAbstracao', 'EntAcontecimento', 'EntCoisa', 'EntLocal', 'EntOrganizacao',
             'EntObra', 'EntOutro', 'EntPessoa', 'EntTempo', 'EntValor',

             'GuiraudR', 'HerdanC', 'HerdanV', 'MedidaK', 'DugastU', 'MaasA', 'HonoresH',

             'PalavrasErroOrtografico', 'HapaxLegomenaLocal', 'PalavrasComunsTam2', 'PalavrasComunsTam3', 'PalavrasComunsTam4',
             'StopWords', 'BRFleshIndex', 'OperadoresLogicos', 'PalavrasCapitalizadas',

             'Author']
        )

    def csv_output(self):
        # TODO: Separate features into syntactical, lexical and so on..
        # 69 features + 1 class
        return "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}," \
               "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}," \
               "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}," \
               "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'{}'".format(

            # Text style features - 10
            round(self.type_token_ratio(), self.ROUNDING_FACTOR),
            round(self.mean_word_len(), self.ROUNDING_FACTOR),
            round(self.mean_sentence_len(), self.ROUNDING_FACTOR),
            round(self.std_sentence_len(), self.ROUNDING_FACTOR),
            round(self.mean_paragraph_len(), self.ROUNDING_FACTOR),
            round(self.std_paragraph_len(), self.ROUNDING_FACTOR),
            len(self.paragraphs) / len(self.text),
            round(self.repeated_words_frequency(), self.ROUNDING_FACTOR),
            self.mean_syllables_per_word(),
            self.monosyllables(),

            # Term count features - 7
            self.term_per_hundred('.'),
            self.term_per_hundred(','),
            self.term_per_hundred('!'),
            self.term_per_hundred(':'),
            self.find_quotes(),
            self.line_breaks_frequency(),
            self.digits_frequency(),

            #POSTAG Features - 24
            self.tag_frequency('ADJ'),
            self.tag_frequency('ADV'),
            self.tag_frequency('ART'),
            self.tag_frequency('N'),
            self.tag_frequency('PREP'),
            self.tag_frequency('PCP'),  # verbo no participio
            self.get_class_frequency_by_start('V'),
            self.get_class_frequency_by_start('K'), #conjunções
            self.get_class_frequency_by_start('PRO'),
            self.get_class_frequency_by_start('PRO')/self.tag_frequency('PREP'), #used in french texts
            self.tag_frequency('notfound'),
            self.get_tags_freq(PortugueseTextualProcessing.CONTENT_TAGS),
            self.get_tags_freq(PortugueseTextualProcessing.FUNCTIONAL_TAGS),
            round(self.noun_phrases(), self.ROUNDING_FACTOR),
            round(self.verb_phrases(), self.ROUNDING_FACTOR),
            self.rich_tags.get_male(),
            self.rich_tags.get_female(),
            self.rich_tags.get_unspecified_gender(),
            self.rich_tags.get_singular(),
            self.rich_tags.get_plural(),
            self.rich_tags.get_first_person(),
            self.rich_tags.get_third_person(),
            self.rich_tags.get_past_tense(),
            self.rich_tags.get_present_tense(),
            self.rich_tags.get_future_tense(),


            #NER Features - 11
            round(len(self.ner_tags) / len(self.tokens), self.ROUNDING_FACTOR),
            self.entity_frequency('ABSTRACCAO'),
            self.entity_frequency('ACONTECIMENTO'),
            self.entity_frequency('COISA'),
            self.entity_frequency('LOCAL'),
            self.entity_frequency('ORGANIZACAO'),
            self.entity_frequency('OBRA'),
            self.entity_frequency('OUTRO'),
            self.entity_frequency('PESSOA'),
            self.entity_frequency('TEMPO'),
            self.entity_frequency('VALOR'),

            # Vocabulary diversity features - 7
            round(self.guiraud_R_measure(), self.ROUNDING_FACTOR),
            round(self.herdan_C_measure(), self.ROUNDING_FACTOR),
            round(self.herdan_V_measure(), self.ROUNDING_FACTOR),
            round(self.K_measure(), self.ROUNDING_FACTOR),
            round(self.dugast_U_measure(), self.ROUNDING_FACTOR),
            round(self.maas_A_measure(), self.ROUNDING_FACTOR),
            round(self.honores_H_measure(), self.ROUNDING_FACTOR),

            # Misc Features - 9
            self.spell_miss_check_frequency(),
            round(self.local_hapax_legommena_frequency(), self.ROUNDING_FACTOR),
            self.collocations_frequency(2),
            self.collocations_frequency(3),
            self.collocations_frequency(4),
            round(self.stop_word_freq(), self.ROUNDING_FACTOR),
            self.flesh_index(),
            self.get_logical_operator_frequency(),
            self.camel_case_frequency(),

            self.author,
        )

    def legacy_features(self):
        """Remove features that are here for future reference"""
        # self.count_characters_frequency(['a']),
        # self.count_characters_frequency(['e']),
        # self.count_characters_frequency(['i']),
        # self.count_characters_frequency(['o']),
        # self.count_characters_frequency(['u']),
        # self.count_consonant_frequency(),
        # self.mean_frequent_word_size(),
        # self.max_word_len(),
        # self.document_len(),
        # round(self.LN_measure(), 8)
        pass
