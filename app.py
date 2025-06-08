import pandas as pd
from bertopic import BERTopic
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

## STEP 1:
## Assign each sentence in the dataset a topic label using BERTopic
df = pd.read_csv("sentiments-cleaned.csv")
texts = df['Text'].dropna().tolist()

topic_model = BERTopic.load("my_stable_topic_model")
topics, probs = topic_model.fit_transform(texts)

df["Topic"] = topics

topic_info = topic_model.get_topic_info()

topic_labels = {
    0: "Experience Review",
    1: "Product Review",
    2: "Website Review"
}

df["Topic Label"] = df["Topic"].map(topic_labels)

## STEP 2:
## Use NLP techniques like 

texts = df['Text'].to_numpy()
labels = df['Topic Label'].to_numpy()


model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

st.title("Topic Label Classifier")

user_input = st.text_input("Enter a sentence:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.write(f"Predicted Topic Label: {prediction}")