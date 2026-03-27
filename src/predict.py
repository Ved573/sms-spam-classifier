import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download once (safe)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# load saved model
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

tfidf = pickle.load(open(os.path.join(BASE_DIR, 'artifact/vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'artifact/model.pkl'), 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# take user input
input_sms = input("Enter message: ")

# transform
transformed_sms = transform_text(input_sms)

# vectorize
vector_input = tfidf.transform([transformed_sms])

# predict
result = model.predict(vector_input)[0]

# output
if result == 'spam':
    print("Spam")
else:
    print("Not Spam")