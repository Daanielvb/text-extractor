import nltk
from nltk import tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta
from pickle import load
from src.util.FileUtil import *


class PortugueseTextualProcessing:
    STOPWORDS = set(nltk.corpus.stopwords.words('portuguese'))
    TAGGER = load(open('t2.pkl', 'rb'))

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
        train = tsents[600:]
        test = tsents[:400]

        t0 = nltk.DefaultTagger('NN')
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


