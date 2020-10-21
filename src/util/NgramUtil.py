from nltk import ngrams


class NgramUtil:

    def __init__(self, df, ngram_sizes, result_sizes):
        """"Initialize Ngram class and perform operations to update internal df"""
        self.df = df
        self.ngram_sizes = ngram_sizes
        self.result_sizes = result_sizes
        self.key_ngrams = self.set_ngrams()
        self.count_freq = self.count_ngram_freq()

    def set_ngrams(self):
        result = dict()
        ngram_list = [self.get_top_ngrams(s, t) for (s, t) in zip(self.ngram_sizes, self.result_sizes)]
        for size, ngram in zip(self.ngram_sizes, ngram_list):
            result[size] = ngram
        return result

    def get_top_ngrams(self, ngram_size, result_size):
        results = dict()
        for text in self.df:
            for grams in ngrams(text.split(), ngram_size):
                if grams in results:
                    results[grams] += 1
                else:
                    results[grams] = 1

        return sorted(results, key=results.get, reverse=True)[:result_size]

    def count_ngram_freq(self):
        count_freq = []
        for text in self.df:
            count = dict()
            for size in self.ngram_sizes:
                count[size] = 0
                for item in self.key_ngrams[size]:
                    count[size] += list(ngrams(text.split(), size)).count(item)
            count_freq.append(count)
        return count_freq

    def upgrade_df_with_count(self, df):
        for size in self.ngram_sizes:
            df['top ' + str(size) + '-gram'] = [i[size]/len(df['Text'].str.split(' ')) for i in self.count_freq]
        return df










