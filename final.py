from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
import re
import string

a=[]

df_fake = pd.read_csv('/home/cycobot/Desktop/project_unmasked/Fake.csv')
df_true = pd.read_csv('/home/cycobot/Desktop/project_unmasked/True.csv')

df_fake.shape, df_true.shape

df_fake["class"] = 0
df_true["class"] = 1

df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_fake.shape, df_true.shape

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

df_marge = pd.concat([df_fake, df_true], axis =0 )
# df_marge.head(10)
df_marge.columns
df = df_marge.drop(["title", "subject","date"], axis = 1)
df.isnull().sum()
df = df.sample(frac = 1)

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)
x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_tcdxsransform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train,y_train)
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
print(len(pred_lr))
for i in pred_lr:
    if(i==0):
        a.append(random.uniform(1,30))
    else:
        a.append(random.uniform(75,99))
print(type(a[0]))
print(len(pred_lr))

# print(pred_lr[0])
# print(classification_report(y_test, pred_lr))
# print(pred_lr['precision'])
