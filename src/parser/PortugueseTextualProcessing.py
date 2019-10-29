import nltk
from nltk import tokenize
from nltk.stem import RSLPStemmer
from nltk.corpus import floresta
import pickle


class PortugueseTextualProcessing:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('portuguese'))
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
        return [w for w in tokenized_text if w not in PortugueseTextualProcessing().stopwords]

    @staticmethod
    def postag(tokenized_text):
        taggger = PortugueseTextualProcessing().retrieve_tagger()
        return [taggger.tag(token) for token in tokenized_text]

    @staticmethod
    def retrieve_tagger():
        #TODO: Fix this method to work with the latest version of NLTK
        tsents = floresta.tagged_sents()
        tsents = [[(w.lower(), PortugueseTextualProcessing().simplify_tag(t)) for (w, t) in sent] for sent in tsents if
                  sent]
        train = tsents[100:]
        test = tsents[:100]

        tagger0 = nltk.DefaultTagger('n')
        nltk.tagger0.accuracy(tagger0, test)

        tagger1 = nltk.UnigramTagger(train, backoff=tagger0)
        nltk.tag.accuracy(tagger1, test)

        tagger2 = nltk.BigramTagger(train, backoff=tagger1)
        nltk.tag.accuracy(tagger2, test)
        return tagger2

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


