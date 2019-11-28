#! /usr/bin/env python

import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
import tensorflow_hub as hub
import os

#from nltk.corpus import stopwords


#SW = set(stopwords.words('english'))
#print(SW)

col = ['Cat', 'SMS']
data = pd.read_csv(r'../Data/SMSSpamCollection', sep='\t', header=None)
data.columns = col
print(data.head())

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
data['Cat'] = Le.fit_transform(data['Cat']) 

print(data.head())

print('start tf')
os.environ["TFHUB_CACHE_DIR"] = r'../cache'

embed = hub.load(r"../cache/f4ea2eb4a9fd72946209ef45271146fae070fb29")

def embedding(text):
    text = [text]
    EmbeddingList = embed(text)['outputs']
    return np.array(EmbeddingList[0])

def clean(text):
    text = text.lower()
    no_punct = [c for c in text if c not in string.punctuation]
    text = ''.join(no_punct)
    return text

data['SMS'] = data['SMS'].apply(lambda x: clean(x))
data['SMSEmbedding'] = data['SMS'].apply(lambda x: embedding(x))

print(data.head())

data.to_json('../Data/embedded.json')