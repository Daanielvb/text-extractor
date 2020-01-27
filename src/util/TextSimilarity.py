import nltk
from collections import Counter
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSimilarity:

    def __init__(self):
        pass

    @staticmethod
    def calculate_jacard_distance(input_text, compare_texts):
        a = set(input_text.split())
        result = ''
        best_dist = 0
        for text in compare_texts:
            b = set(text.split())
            c = a.intersection(b)
            distance = float(len(c)) / (len(a) + len(b) - len(c))
            if distance > best_dist:
                best_dist = distance
                result = text
        return result

    @staticmethod
    def get_vectors(text, vectorizer):
        return vectorizer.transform(text).toarray()

    @staticmethod
    def get_cosine_sim(input_text, compare_texts):
        # TODO: Improve this method
        compare_texts.append(input_text)
        vectorizer = CountVectorizer(compare_texts)
        vectorizer.fit(compare_texts)
        vectors = [t for t in TextSimilarity().get_vectors(compare_texts, vectorizer)]
        input_vector = vectors.pop(0)
        distances = [1 - spatial.distance.cosine(input_vector, vect) for vect in vectors]
        return distances.index(max(distances))


if __name__ == '__main__':

    X = ['meu nome é camila', 'daniel joga futebol', 'cansei se ser o ultimo']
    jaccard = TextSimilarity().calculate_jacard_distance('meu nome é daniel e eu gosto muito de jogar futebol', X)
    cosine_idx = TextSimilarity().get_cosine_sim('meu nome é daniel e eu gosto muito de jogar futebol', X)
    print (jaccard)
    print(X[cosine_idx])


