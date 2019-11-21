import nltk
from nltk import tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta
from pickle import load
from src.util.FileUtil import *
import numpy as np
import spatial
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
    def postag(tokenized_text):
        result = []
        for token in tokenized_text:
            result.append(PortugueseTextualProcessing().TAGGER.tag([token.lower()]))
        return result

    @staticmethod
    def build_tagger():
        tsents = floresta.tagged_sents()
        tsents = [[(w.lower(), PortugueseTextualProcessing().simplify_tag(t)) for (w, t) in sent] for sent in tsents if
                  sent]
        train = tsents[:6000]
        test = tsents[6000:]

        print(len(train))
        print(len(test))
        t0 = nltk.DefaultTagger('notfound')
        t1 = nltk.UnigramTagger(train, backoff=t0)
        t2 = nltk.BigramTagger(test, backoff=t1)
        print(t0.evaluate(test))
        print(t1.evaluate(test))
        print(t2.evaluate(test))
        FileUtil.write_pickle_file('pttag.pkl', t2)
        return t2

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
    def load_vector():
        word_embedding = {}
        with open("glove_s100.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                word_embedding[word] = vector
        print('Found %s word vectors.' % len(word_embedding))
        return word_embedding

    @staticmethod
    def load_vector_2():
        word_embedding = {}
        with open("glove_s100.txt", 'r') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                word_embedding[word] = coefs
        print('Found %s word vectors.' % len(word_embedding))
        return word_embedding

    @staticmethod
    def find_closest_embeddings(embedding):
        return sorted(PortugueseTextualProcessing.WORD_EMBEDDINGS.keys(),
                      key=lambda word: spatial.distance.euclidean(PortugueseTextualProcessing.WORD_EMBEDDINGS[word], embedding))

    @staticmethod
    def build_embedding_matrix(word_embedding, input_text):
        tokenizer = Tokenizer(num_words=PortugueseTextualProcessing.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(input_text)
        # sequences = tokenizer.texts_to_sequences(input_text)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        num_words = min(PortugueseTextualProcessing.MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, PortugueseTextualProcessing.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= PortugueseTextualProcessing.MAX_NUM_WORDS:
                continue
            embedding_vector = word_embedding.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector[:PortugueseTextualProcessing.EMBEDDING_DIM]
        return embedding_matrix
