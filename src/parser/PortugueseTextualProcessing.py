import nltk
from nltk import tokenize, ne_chunk
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta, mac_morpho, masc_tagged
from pickle import load

from src.model.RichTagFrequency import *
from src.util.FileUtil import *
import numpy as np
import spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import RegexpParser, Tree
from nltk.corpus import wordnet as wn
from nltk import RegexpParser
from nltk.stem import SnowballStemmer
import pyphen
from collections import defaultdict
from src.parser.syllable.Silva2011SS import *
from spacy.matcher import Matcher
from spacy.util import filter_spans
import pt_core_news_sm


class PortugueseTextualProcessing:
    NLP = pt_core_news_sm.load()
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    CUSTOM_STOPWORDS = FileUtil().get_words_from_file('../resources/custom_stopwords.txt')
    TAGGER = load(open('pttag-mm.pkl', 'rb'))
    EMBEDDING_DIM = 100
    MAX_NUM_WORDS = 20000
    LONG_SENTENCE_SIZE = 12
    SHORT_SENTENCE_SIZE = 6
    PT_DICT = pyphen.Pyphen(lang='pt_BR')
    SILVA_SYLLABLE_SEPARATOR = Silva2011SyllableSeparator()
    NER_PT_TAGGER = FileUtil().load_ner_pickle()
    NER_TIME_TAGGER = FileUtil().load_ner_pickle('../resources/cat-entropy_cutoff_0.08.pickle')
    LOGICAL_OPERATORS = ['e', 'nada', 'a menos que', 'ou', 'nunca', 'sem que', 'não', 'jamais', 'nem'
                         'caso', 'se', 'nenhum', 'nenhuma', 'então é porque', 'desde que', 'contanto que',
                         'uma vez que', 'fosse']
    CONTENT_TAGS = ['N', 'ADJ', 'ADV', 'V']
    FUNCTIONAL_TAGS = ['ART', 'PREP', 'PRON','K']
    DEFAULT_NOT_FOUND_TAG = 'notfound'
    CASE_SENSITIVE_PATTERN = '[A-Z][a-z]*'
    NUMBER_ONLY_PATTERN = '[0-9]'
    RICH_TAG_TYPES = ['Gender', 'Number', 'Person', 'PronType', 'VerbForm', 'Tense']

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
        tokens = tokenize.word_tokenize(text, language='portuguese')
        slash_tokens = [i for i in tokens if '/' in i]
        if slash_tokens:
            PortugueseTextualProcessing().separate_slash_tokens(tokens, slash_tokens)
        return tokens

    @staticmethod
    def count_lemmas(text):
        doc = PortugueseTextualProcessing.NLP(text)
        return len([token for token in doc if token.text != token.lemma_])

    @staticmethod
    def get_rich_tags(text):
        doc = PortugueseTextualProcessing.NLP(text)
        tags = [(token.lemma_, token.pos_, token.tag_) for token in doc]
        tagged_text = ''.join([tag[2] for tag in tags if "|" in tag[2]]).split("|")
        return PortugueseTextualProcessing().extract_tags(tagged_text)

    @staticmethod
    def extract_tags(tagged_text):
        tags = []
        for tag in PortugueseTextualProcessing().RICH_TAG_TYPES:
            if tag != 'Person':
                type_tags = PortugueseTextualProcessing().get_regular_tags(tag,
                                                                           PortugueseTextualProcessing().CASE_SENSITIVE_PATTERN, tagged_text)
            else:
                type_tags = PortugueseTextualProcessing().get_regular_tags(tag,
                                                                           PortugueseTextualProcessing().NUMBER_ONLY_PATTERN,
                                                                           tagged_text)
            tags.append(RichTagFrequency(tag, ''.join(type_tags).replace(tag + '=', ' ').split(' ')[1:]))

        return tags

    @staticmethod
    def get_regular_tags(pattern, case, tagged_text):
        return re.findall(pattern + '=' + case, ''.join(tagged_text))

    @staticmethod
    def separate_slash_tokens(all_tokens, slash_text):
        """Algorithm that converts all words with a slash in the middle in two separate words
        eg: national research council/national academy of sciences -> national research council national
        academy of sciences.
        eg: This is my current website/ -> This is my current website
        """
        for idx, token in enumerate(all_tokens):
            if token in slash_text:
                is_first = True
                for part in token.split("/"):
                    if part != '':
                        if is_first:
                            all_tokens[idx] = part
                            is_first = False
                        else:
                            all_tokens.insert(idx + 1, part)

    @staticmethod
    def stem(tokenized_text):
        """Warning: This stem is currently creating a lot of nonexistent words, do not use yet!"""
        return [RSLPStemmer().stem(token) for token in tokenized_text]

    @staticmethod
    def another_stem(tokenized_text):
        return [SnowballStemmer("portuguese").stem(token) for token in tokenized_text]

    @staticmethod
    def remove_stopwords(tokenized_text):
        return [w for w in tokenized_text if w not in PortugueseTextualProcessing().STOPWORDS]

    @staticmethod
    def postag(tokenized_text, as_list=True):
        result = []
        for token in tokenized_text:
            if as_list:
                tag = PortugueseTextualProcessing().TAGGER.tag([token.lower()])
                result.append([(tag[0][0], 'PREP')]) if tag[0][1] == '' else result.append(tag)
            else:
                tag = PortugueseTextualProcessing().TAGGER.tag([token.lower()])[0]
                result.append((tag[0], 'PREP')) if tag[1] == '' else result.append(tag)
        return result

    @staticmethod
    def ner_chunks(tokens):
        tagged_text = [i for i in PortugueseTextualProcessing.postag(tokens, as_list=False) if i[1] != PortugueseTextualProcessing().DEFAULT_NOT_FOUND_TAG]
        chunked = PortugueseTextualProcessing().NER_PT_TAGGER.parse(tagged_text)
        trees = [i for i in chunked if type(i) == Tree]
        entities = [i.label() for i in trees]
        # TODO: Remove OBRA with 1 token just punct (,) or ART, COISA with 1 token just punct (,), LOCAL 1 NUM and
        # check cases for ORGANIZACAO
        # for i in trees:
        #     print(i)
        chunks = PortugueseTextualProcessing().extract_chunks(chunked)
        return entities

    @staticmethod
    def time_chunks(tagged_text):
        chunked = PortugueseTextualProcessing().NER_TIME_TAGGER.parse(tagged_text)
        trees = [i for i in chunked if type(i) == Tree and i.label() == 'TEMPO']
        # TODO: Implement rule to remove cases with only one KC or NUM with 1 digit
        return trees
        
    @staticmethod
    def extract_chunks(chunked):
        continuous_chunk = []
        current_chunk = []
        for subtree in chunked:
            if type(subtree) == Tree:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk

    @staticmethod
    def get_number_of_noun_phrases(tokenized_text):
        """ text = "João comprou um carro esportivo
        tokens = nltk.word_tokenize(text)
        tagged = tagger.tag(tokens)
        gramatica = 'rPADRAO: {<N><ADJ>+}
        analiseGramatical = nltk.RegexpParser(gramatica)
        analiseGramatical.parse(tagged)
        (S João/NPROP comprou/V um/ART (PADRAO carro/N esportivo/ADJ))
        Modifiers per Noun Phrase	0.248
        Noun Phrase Incidence	445.960
        Words before Main Verb	1.750

        Determinante(ART|PROADJ) | Nome(N) | opcional Elemento modificador (adj)
        ART|N|PRP
        """
        tag_string = ' '.join([tag[1] for tag
                               in PortugueseTextualProcessing.postag(tokenized_text, as_list=False)
                               if tag[1] != PortugueseTextualProcessing.DEFAULT_NOT_FOUND_TAG])
        # NP = "NP: {(<V\w+>|<N\w?>)+.*<N\w?>}"
        #'(ART|PROADJ) ADJ\w+ N'
        #np_rgx = '(V\w+|N\w?) \w+ N'
        # An optional determiner (DT), zero or more adjectives (JJ), and a noun (NN), proper noun (NP), or pronoun (PRN)
        np_rgx = "((?:ART))? (?:\w+ ADJ) * +(?:N|NPROP|PROADJ) "
        matches = re.findall(np_rgx, tag_string)
        return len(matches)

    @staticmethod
    def get_continuous_chunks(tokenized_text):
        # this regex is not working, change to another later
        NP = "(?:(?:\w+ ART)?(?:\w+ ADJ) *)?\w + (?:N[NP] | PRN)"
        chunker = RegexpParser(NP)       
        tagged_text = PortugueseTextualProcessing.postag(tokenized_text, as_list=False)
        chunked = chunker.parse(tagged_text)
        return PortugueseTextualProcessing().extract_chunks(chunked)

    @staticmethod
    def break_in_syllables(word):
        """"Silva algorithm does not work very well with ss and rr syllables"""
        if 'ss' in word or 'rr' in word:
            return PortugueseTextualProcessing().PT_DICT.inserted(word).split('-')
        return [s for s in PortugueseTextualProcessing().SILVA_SYLLABLE_SEPARATOR.separate(word) if s != '']

    @staticmethod
    def get_syllable_counts(tokens):
        words = defaultdict()
        count = 0
        for token in tokens:
            if token not in words.keys():
                words[token] = []
            syllable_count = len(PortugueseTextualProcessing().break_in_syllables(token))
            count += syllable_count
            words[token].append(syllable_count)
        return words, count

    @staticmethod
    def get_monosyllable_counts(tokens):
        words, count = PortugueseTextualProcessing.get_syllable_counts(tokens)
        return len([w for w in words.items() if w[1][0] == 1])

    @staticmethod
    def get_ptBR_flesch_index(tokens, phrases):
        """ILF = 248.835 – (1.015 x ASL) – (84.6 x ASW)
         ASL é o número de palavras dividido pelo número de sentenças e ASW é o
        número de sílabas dividido pelo número de palavras
        """
        index = 248.835 - (1.015 * (PortugueseTextualProcessing().words_per_phrases(tokens, phrases)) - \
                (84.6 * (PortugueseTextualProcessing().syllables_per_word(tokens))))
        return index, PortugueseTextualProcessing().get_readability_level(index)

    @staticmethod
    def words_per_phrases(tokens, phrases):
        return len(phrases)/len(tokens)

    @staticmethod
    def syllables_per_word(tokens):
        _, syllable_count = PortugueseTextualProcessing().get_syllable_counts(tokens)
        return syllable_count / len(tokens)

    @staticmethod
    def get_readability_level(index):
        if index < 25:
            return "Muito fácil - Ensino superior"
        elif 25 < index < 50:
            return "Díficil - Ensino médio"
        elif 50 < index < 75:
            return "Fácil - 6 a 9 ano"
        else:
            return "Muito fácil - 1 a 5o ano"

    @staticmethod
    def get_number_of_verb_phrases(text):
        """A verb phrase consists of an auxiliary, or helping, verb and a main verb. The helping verb always precedes the main verb
        Some sentences will feature a subject or a modifier placed in between a verb phrase’s helping and main verbs.
        Note that the subject or modifier is not considered part of the verb phrase.
        https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy
        """
        matcher = Matcher(PortugueseTextualProcessing.NLP.vocab)

        pattern = [{'POS': 'VERB', 'OP': '?'},
                   {'POS': 'ADP', 'OP': '*'},
                   {'POS': 'VERB', 'OP': '+'}]
        matcher.add("Verb phrase", None, pattern)
        doc = PortugueseTextualProcessing.NLP(text)
        return matcher(doc)

    def num_words_before_main_verb(self):
        pass

    @staticmethod
    def build_tagger(corpus=mac_morpho, tagger_name='pttag-mm.pkl'):
        tsents = corpus.tagged_sents()
        tsents = [[(w.lower(), PortugueseTextualProcessing().simplify_tag(t)) for (w, t) in sent] for sent in tsents if
                  sent]
        train = tsents[:35000]
        test = tsents[35000:]

        print(len(train))
        print(len(test))
        t0 = nltk.DefaultTagger(PortugueseTextualProcessing().DEFAULT_NOT_FOUND_TAG)
        t1 = nltk.UnigramTagger(train, backoff=t0)
        t2 = nltk.BigramTagger(test, backoff=t1)
        t3 = nltk.TrigramTagger(test, backoff=t2)
        print(t0.evaluate(test))
        print(t1.evaluate(test))
        print(t2.evaluate(test))
        print(t3.evaluate(test))
        FileUtil.write_pickle_file(tagger_name, t3)
        return t3

    @staticmethod
    def simplify_tag(t):
        if "+" in t:
            return t[t.index("+") + 1:]
        else:
            return t

    @staticmethod
    def concordance(word, context=30):
        print(f"palavra: {word}, contexto: {context} caracteres")
        for sent in floresta.sents():
            if word in sent:
                pos = sent.index(word)
                left = " ".join(sent[:pos])
                right = " ".join(sent[pos + 1:])
                print(f"{left[-context:]} '{word}' {right[:context]}")

    @staticmethod
    def load_vector(vocabulary_tokenizer, embedding_file='glove_s100.txt'):
        word_embedding = {}
        vocabulary_tokenizer
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                if word in vocabulary_tokenizer.word_index:
                    coefs = np.fromstring(coefs, 'f', sep=' ')
                    word_embedding[word] = coefs
        print('Found %s word vectors.' % len(word_embedding))
        return word_embedding

    @staticmethod
    def find_closest_embeddings(embedding):
        return sorted(PortugueseTextualProcessing.WORD_EMBEDDINGS.keys(),
                      key=lambda word: spatial.distance.euclidean(PortugueseTextualProcessing.WORD_EMBEDDINGS[word], embedding))

    @staticmethod
    def convert_corpus_to_number(dataframe):
        """:param dataframe pandas.Dataframe"""
        corpus = dataframe['Text']
        tokenizer = Tokenizer(num_words=PortugueseTextualProcessing.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(corpus)

        embedded_sentences = tokenizer.texts_to_sequences(corpus)
        print(embedded_sentences)

        longest_sentence = max(corpus, key=lambda sentence: len(nltk.word_tokenize(sentence, language='portuguese')))
        max_sentence_len = len(nltk.word_tokenize(longest_sentence, language='portuguese'))

        padded_sentences = pad_sequences(embedded_sentences, max_sentence_len, padding='post')

        return tokenizer, padded_sentences, max_sentence_len

    @staticmethod
    def build_embedding_matrix(word_embedding_dict, vocab_length, tokenizer, size=100):
        embedding_matrix = np.zeros((vocab_length, size))
        for word, index in tokenizer.word_index.items():
            embedding_vector = word_embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector[:size]
        return embedding_matrix

    @staticmethod
    def generate_stopwords_df(df):
        # TODO: Improve this method to avoid removing all paragraphs and line space by the join
        df['Text_without_stopwords'] = df['Text'].apply(
            lambda x: ' '.join(
                [word for word in x.split(' ') if word.lower() not in (PortugueseTextualProcessing.STOPWORDS)]))
        df['Text_without_stopwords'] = df['Text_without_stopwords'].apply(
            lambda x: ' '.join(
                [word for word in x.split(' ') if word.lower() not in (PortugueseTextualProcessing.CUSTOM_STOPWORDS)]))
        df['Text'] = df['Text_without_stopwords']
        df.pop('Text_without_stopwords')
        return df
