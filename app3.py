from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import re
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import multiprocessing
cores = multiprocessing.cpu_count()

model=gensim.models.Word2Vec( window=11,
                              min_count=2,
                              vector_size=300,
                              alpha=0.025,
                              negative=20,
                              workers=cores-1,
                              sg=0
                              )

stopwords_set = set(stopwords.words('english'))

text=""
text1=""
filename="sentiment_model2.pkl"
filename1="vectorizer_model.pkl"

with open(filename1, 'rb') as file:
    vectorizer = pickle.load(file)

def display_sarcastic_remark(remark):
    st.title(remark)
    time.sleep(0.1)

st.header('Sentiment Analysis')
with st.title('Analyze Text'):
	text = st.text_input('Text here: ')
if text:
	text1=text
	blob = TextBlob(text)
#st.write('Polarity: ', round(blob.sentiment.polarity,2))
#st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
plt.figure(figsize = (20,20))

if(text1!=""):
    st.title("Cleaned Text")
    text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
    tknzr = TweetTokenizer(strip_handles=True)
    text1=tknzr.tokenize(text1)
    text1=str(text1)
    text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
    text1=cleantext.clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
    st.write(text1)

with open(filename, 'rb') as file:
    model = pickle.load(file)
unseen_tweets=[text]
unseen_df=pd.DataFrame(unseen_tweets)
unseen_df.columns=["Unseen"]

X_test = vectorizer.transform(unseen_tweets)
y_pred = model.predict(X_test)

if text!="":
    if(y_pred==0):
        remark = "That's Figurative!😄"
        display_sarcastic_remark(remark)
    if(y_pred==1):
        remark = "That's Irony!😏"
        display_sarcastic_remark(remark)
    if(y_pred==2):
        remark = "That's Regular!😐"
        display_sarcastic_remark(remark)
    if(y_pred==3):
        remark = "That's Sarcasm!🙃"
        display_sarcastic_remark(remark)
else:
    st.write(text1)
    remark = "No Words to Analyze"
    display_sarcastic_remark(remark)