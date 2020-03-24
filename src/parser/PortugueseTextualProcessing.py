import nltk
from nltk import tokenize, ne_chunk
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta, mac_morpho, masc_tagged
from pickle import load
from src.util.FileUtil import *
import numpy as np
import spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import RegexpParser, Tree
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer
import pyphen
from collections import defaultdict
from src.parser.syllable.Silva2011SS import *


class PortugueseTextualProcessing:
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    CUSTOM_STOPWORDS = FileUtil().get_words_from_file('custom_stopwords.txt')
    TAGGER = load(open('pttag-mm.pkl', 'rb'))
    EMBEDDING_DIM = 100
    MAX_NUM_WORDS = 20000
    PT_DICT = pyphen.Pyphen(lang='pt_BR')
    SILVA_SYLLABLE_SEPARATOR = Silva2011SyllableSeparator()
    LOGICAL_OPERATORS = ['e', 'nada', 'a menos que', 'ou', 'nunca', 'sem que', 'não', 'jamais', 'nem'
                         'caso', 'se', 'nenhum', 'nenhuma', 'então é porque', 'desde que', 'contanto que',
                         'uma vez que', 'fosse']
    CONTENT_TAGS = ['N', 'ADJ', 'ADV', 'V']
    FUNCTIONAL_TAGS = ['ART', 'PREP', 'PRON','K']

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
        return tokenize.word_tokenize(text, language='portuguese')

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
                result.append(PortugueseTextualProcessing().TAGGER.tag([token.lower()]))
            else:
                result.append(PortugueseTextualProcessing().TAGGER.tag([token.lower()])[0])
        return result

    @staticmethod
    def get_number_of_noun_phrases(tokenized_text):
        # TODO : Improve this method
        """ text = "João comprou um carro esportivo
        tokens = nltk.word_tokenize(text)
        tagged = tagger.tag(tokens)
        gramatica = 'rPADRAO: {<N><ADJ>+}
        analiseGramatical = nltk.RegexpParser(gramatica)
        analiseGramatical.parse(tagged)
        (S João/NPROP comprou/V um/ART (PADRAO carro/N esportivo/ADJ))"""
        tag_string = ' '.join([tag[1] for tag in PortugueseTextualProcessing.postag(tokenized_text, as_list=False)])
        # NP = "NP: {(<V\w+>|<N\w?>)+.*<N\w?>}"
        #'(ART|PROADJ) ADJ\w+ N'
        np_rgx = '(V\w+|N\w?) \w+ N'
        matches = re.findall(np_rgx, tag_string)
        return len(matches)

    @staticmethod
    def break_in_syllables(word):
        #print(word)
        # TODO: Find a strategy for better usage on syllables (size/tag)
        #silva = [s for s in PortugueseTextualProcessing().SILVA_SYLLABLE_SEPARATOR.separate(word) if s != '']
        pt_dic = PortugueseTextualProcessing().PT_DICT.inserted(word).split('-')
        #print('silva: '+ str(silva))
        #print('ptdoc: '+ str(pt_dic))
        return pt_dic

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
    #TODO: Investigate: https://realpython.com/natural-language-processing-spacy-python/#verb-phrase-detection
    def get_number_of_verb_phrases(tokenized_text):
        """A verb phrase consists of an auxiliary, or helping, verb and a main verb. The helping verb always precedes the main verb
        Some sentences will feature a subject or a modifier placed in between a verb phrase’s helping and main verbs.
        Note that the subject or modifier is not considered part of the verb phrase."""
        tag_string = ' '.join([tag[1] for tag in PortugueseTextualProcessing.postag(tokenized_text, as_list=False)])
        #V NP | V NP PP

        vp_rgx = 'V\w+ ADV*V+'
        matches = re.findall(vp_rgx, tag_string)
        return len(matches)

    # TODO
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
        t0 = nltk.DefaultTagger('notfound')
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
        df['Text_without_stopwords'] = df['Text'].apply(
            lambda x: ' '.join(
                [word for word in x.split() if word.lower() not in (PortugueseTextualProcessing.STOPWORDS)]))
        df['Text_without_stopwords'] = df['Text_without_stopwords'].apply(
            lambda x: ' '.join(
                [word for word in x.split() if word.lower() not in (PortugueseTextualProcessing.CUSTOM_STOPWORDS)]))
        df['Text'] = df['Text_without_stopwords']
        df.pop('Text_without_stopwords')
        return df
