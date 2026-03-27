import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# setup
ps = PorterStemmer()

BASE_DIR = os.path.dirname(__file__)

tfidf = pickle.load(open(os.path.join(BASE_DIR, 'artifact/vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'artifact/model.pkl'), 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 'spam':
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")