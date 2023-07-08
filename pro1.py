import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('/Users/ekamsinghahuja/Desktop/machine learning cc/projects/vectorizer.pkl','rb'))
model = pickle.load(open('/Users/ekamsinghahuja/Desktop/machine learning cc/projects/model.pkl','rb'))
st.title("Email/SMS spam classifier")
input_sms = st.text_input('enter the message')

#preprocess
transform_sms = transform_text(input_sms)
#vectorise
vec = tfidf.transform([transform_sms]) 
#predict
result = model.predict(vec)[0]
#display
if result==1:
    st.header("spam")
else:
    st.header("not spam")
