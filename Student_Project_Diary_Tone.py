import streamlit as st
import re
import nltk
import re
import nltk
from textblob import TextBlob
import glob
from pathlib import Path
import os
import pandas as pd
import plotly.express as px

from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer

filepaths = glob.glob("diary_files/*txt")

alldiaries = []
dates = []

for filepath in filepaths:
    with open(filepath, "r") as file:
        text = file.read()
        alldiaries.append(text)
        date = Path(filepath).stem
        dates.append(date)

analyzer = SentimentIntensityAnalyzer()
vaderscores = {}
textblobscores = {}
for nr, entry in enumerate(alldiaries):
    vaderscore = analyzer.polarity_scores(entry)
    vaderscores[nr] = vaderscore

vaderframe = pd.DataFrame.from_dict(vaderscores)

textblobscores = {}
for nr, entry in enumerate(alldiaries):
    textscore = TextBlob(entry).sentiment
    textblobscores[nr] = textscore

textframe = pd.DataFrame.from_dict(textblobscores)
textlist = textframe.iloc[0].tolist()
vaderlist = vaderframe.iloc[3].tolist()

figure = px.line(x=dates, y=vaderlist, title="VADER Scores for selected Diary Entries", labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by VADER"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure)
st.write(vaderlist)

figure2 = px.line(x=dates, y=textlist, title="TextBlob Scores for selected Diary Entries",
                                             labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by TextBlob.sentiment",
                                               "title": "TEST TITLE"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure2)
st.write(textlist)

# """Add this to here:
# import tensorflow as tf
# from transformers import pipeline
#
# sentiment_analyzer = pipeline("sentiment-analysis")
#
# tensorscores = {}
# for nr, entry in enumerate(alldiaries):
#     tensorscore = sentiment_analyzer(entry)
#     tensorscores[nr] = tensorscore
#
# tensorframe = pd.DataFrame.from_dict(tensorscores)
# tensorlist = tensorframe.iloc[!!!!!IDK!!!!!].tolist()
# """

tensorscores = [-1, 1, -1, 1, 1, -1, -.8]

bingsinstructions = """ These instructions were presented to the Bing Chat AI 
2023-07-07 at approximately 1 PM EDT, with the AI set to conversational mode.
The exact same prompts were then provided to chatgpt set to the free 3.5 mode, and Bard. 
The instructions are:

I am about to present you with a selection of text. I would like you to evaluate the sentiment of this text and give the text a sentiment score on range from -1, the most negative possible sentiment, to 1, the most positive possible sentiment. A selection of text such as "I hate myself and everything else, life is terrible :(" should receive a score of -1, while a selection of text such as "I love myself and the entire world, everything about life is beautiful!" should receive a score of 1. A selection of text such as "I had fries today." should receive a score of 0, for a perfectly neutral sentinment. You can and should assign decimals to your sentiment score, if appropriate; a sentence such as "I had some pretty damn good fries today!" is not a 0, but it's not a perfect 1 either, so should receive a score such as, for example, 0.65, while "Man, fries are so EXPENSIVE nowdays" should receive a negative sentiment score, but not a perfect -1. 

Those are my instructions. Please return a sentiment score for the text below according to these instructions, and if possible, explain your reasoning.  Do not evaluate the sentiment of the instructions, only the text below the instructions. The instructions end here.

Subsequent to the first entry, each instruction was also prefaced with "Thank you, you did a great job!" The instructions were then repeated on a new line.
"""

bingscores = [-.8, 0.85, -0.2, 0.4, 0.1, -0.75, 0.55]
chatgpt3_5scores = [-0.75, 0.9, 0.3, 0.75, 0.4, -0.8, 0.6]
bardscores = [(-0.67, -0.75, -0.75),  (0.75, 0.85, 0.75), (-0.4, -0.3, -0.3),
              (0.7, 0.75, 0.7), (0.3, 0.4, 0.3), (-0.8, -0.8, -0.7), (0.4, 0.5, 0.4)]
bardmeans = [-0.723, 0.783, -0.333, 0.7166, 0.333, -0.766, 0.433]

figure3 = px.line(x=dates, y=bingscores, title="Bing Conversational GPT 4.0 Scores for selected Diary Entries",
                                             labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by TextBlob.sentiment",
                                               "title": "TEST TITLE"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure3)
st.write(bingscores)

figure4 = px.line(x=dates, y=chatgpt3_5scores, title="ChatGPT3.5 Scores for selected Diary Entries",
                                             labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by TextBlob.sentiment",
                                               "title": "TEST TITLE"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure4)
st.write(chatgpt3_5scores)

figure5 = px.line(x=dates, y=bardmeans, title="Google Bard Scores for selected Diary Entries",
                                             labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by TextBlob.sentiment",
                                               "title": "TEST TITLE"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure5)
st.write(bardmeans)

figure6 = px.line(x=dates, y=tensorscores, title="Huggingface Pipeline Scores for selected Diary Entries using TensorFlow",
                                             labels={"x": "Dates",
                                                  "y": "Sentiment Scores as determined by TextBlob.sentiment",
                                               "title": "TEST TITLE"})  # Notice labels accepts a DICTIONARY as its input.
st.plotly_chart(figure6)
st.write(tensorscores)

st.info(bingsinstructions)