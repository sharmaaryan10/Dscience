import streamlit as st
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline 
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title('News Classification')
df = pd.read_csv('cleaned_news.csv')
df 

# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()


vectorizer = TfidfVectorizer(stop_words="english")
X = df['cleaned']
Y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression(random_state=1))])


# #Training model
model = pipeline.fit(X_train, y_train)


# for testing

# 1. opening a file 
# file = open('news.txt','r' , encoding = 'latin1')
# news = file.read()
# file.close()

# 2. taking input from user 
news = st.text_area("Text to translate")
if st.button("Submit"):
    # news = input("Enter news = ")
    news_data = {'predict_news':[news]}
    news_data_df = pd.DataFrame(news_data)

    predict_news_cat = model.predict(news_data_df['predict_news'])
    st.write("Predicted news category = ",predict_news_cat[0])
