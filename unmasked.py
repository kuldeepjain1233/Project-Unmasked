from nltk.corpus.reader.chasen import test
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# nltk.download('stopwords')

news_dataset = pd.read_csv("/home/cycobot/Downloads/Datasets/train.csv")
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' '+ news_dataset['title']

x = news_dataset.drop(columns='label', axis=1)
y = news_dataset['label']
print(y)

port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word)for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

x = news_dataset['content'].values
y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,stratify=y, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print(test_data_accuracy)
# print(x_test)
print(y_test)