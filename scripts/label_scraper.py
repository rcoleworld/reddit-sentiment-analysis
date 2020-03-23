import pandas as pd
import collections


TRAIN_DATA = '../train_data.csv'

df =  pd.read_csv(TRAIN_DATA)
sentiments = df['sentiment']

sentiment_freq = collections.Counter(sentiments)
sentiment_set = set(sentiments)

print(sentiment_set)