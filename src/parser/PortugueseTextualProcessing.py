import nltk
from nltk import tokenize, ne_chunk
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta, genesis, machado, mac_morpho
from pickle import load
from src.util.FileUtil import *
import numpy as np
import spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import RegexpParser, Tree


class PortugueseTextualProcessing:
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    TAGGER = load(open('pttag.pkl', 'rb'))
    EMBEDDING_DIM = 100
    MAX_NUM_WORDS = 20000

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text):
        return tokenize.word_tokenize(text, language='portuguese')

    @staticmethod
    def stem(tokenized_text):
        """This stem is currently creating a lot of nonexistent words"""
        return [RSLPStemmer().stem(token) for token in tokenized_text]

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
    def get_continuous_chunks(tokenized_text, chunk_func=ne_chunk):
        # Defining a grammar & Parser
        NP = "NP: {(<(v-fin|v-inf|v-pcp|v-ger)\w+>|<n\w?>)+.*<n\w?>}"
        chunker = RegexpParser(NP)
        tagged = PortugueseTextualProcessing.postag(tokenized_text, as_list=False)
        cs = chunker.parse(tagged)
        continuous_chunk = []
        current_chunk = []

        for subtree in cs:
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
    def build_tagger():
        tsents = []
        tsents.extend(floresta.tagged_sents())
        tsents.extend(mac_morpho.tagged_sents())
        tsents = [[(w.lower(), PortugueseTextualProcessing().simplify_tag(nltk.pos_tag(t))) for (w, t) in sent] for sent in tsents if
                  sent]
        train = tsents[:6000]
        test = tsents[6000:]

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
        FileUtil.write_pickle_file('pttag.pkl', t3)
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