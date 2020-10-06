

class GlobalFeatures:

    @staticmethod
    def calculate_global_hapax_legomena(df, column_name='Global hapax'):
        results = []
        for idx, text in enumerate(df['Text']):
            excluding_df = df.drop(idx)
            unique_words = set(text.split(' '))
            vocabulary = list(set(' '.join(excluding_df['Text']).split(' ')))
            results.append((len(unique_words) - sum(el in list(unique_words) for el in vocabulary)) / len(text))
        df[column_name] = results
        return df
