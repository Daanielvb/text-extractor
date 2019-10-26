import nltk
from nltk import tokenize
from nltk.stem.RSLPStemmer import *


class PortugueseTextualProcessing:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        pass

    @staticmethod
    def tokenize(text):
        return tokenize.word_tokenize(text, language='portuguese')

    @staticmethod
    def stem(tokenized_text):
        return nltk.stem.RSLPStemmer().stem(tokenized_text)

    @staticmethod
    def remove_stopwords(tokenized_text):
        return [w for w in tokenized_text if not w in PortugueseTextualProcessing().stop_words]

    @staticmethod
    def postag(tokenized_text):
        # TODO: See if this tagger works fine: https://github.com/fmaruki/Nltk-Tagger-Portuguese/
        pass




