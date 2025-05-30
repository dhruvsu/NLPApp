import pandas as pd
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sentiment-analysis.csv", quotechar='"')
df = df.dropna()

new_df = df["Text, Sentiment, Source, Date/Time, User ID, Location, Confidence Score"].str.extract(r'^"([^"]+)",\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(.+)$')
new_df.columns = ["Text", "Sentiment", "Source", "Date/Time", "User ID", "Location", "Confidence Score"]

texts = new_df['Text'].to_numpy()
labels = new_df['Sentiment'].to_numpy()

model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

user_input = st.text_input("Enter a sentence:")

if user_input:
    prediction = model.predict([user_input])[0]
    sentiment = "Positive" if prediction == "Positive" else "Negative"
    
    st.write(f"Predicted Sentiment: {sentiment}")