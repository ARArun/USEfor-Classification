#! /usr/bin/python3

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords


SW = set(stopwords.words('english'))
#print(SW)

col = ['Cat', 'SMS']
data = pd.read_csv(r'../Data/SMSSpamCollection', sep='\t', header=None)
data.columns = col
print(data.head())

from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
data['Cat'] = Le.fit_transform(data['Cat']) 

print(data.head())

