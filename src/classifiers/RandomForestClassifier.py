import pandas as pd
from sklearn.preprocessing import LabelEncoder

class RandomForestClassifier:

    def __init__(self):
        pass

    def pre_process(self, df):
        # show_column_distribution(df, 'Author')

        y = df.pop('Author')

        le = LabelEncoder()
        le.fit(y)
        encoded_Y = le.transform(y)
        #save_encoder(le)
