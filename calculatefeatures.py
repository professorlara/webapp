
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from transformers import pipeline

# Define the functions
def huggingface(text,category):
     # Initialize the sentiment analysis pipeline with the RoBERTa model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Split the text into lines and filter out empty lines and lines starting with '['
    lines = [line for line in text.split('\n') if line and not line.startswith('[')]

    # Analyze sentiment for each line
    sentiments = sentiment_pipeline(lines)

    # Initialize counters and lists for storing results
    count_negative = 0
    count_positive = 0
    sentiment_scores = []

    # Process the sentiment results
    for sentiment in sentiments:
        if sentiment['label'] == 'NEGATIVE':
            count_negative += 1
            score = sentiment['score'] * -1
        else:
            count_positive += 1
            score = sentiment['score']
        sentiment_scores.append(score)

    # Calculate sentiment statistics
    if sentiment_scores:
        features = {
            'average_sentiment': np.mean(sentiment_scores),
            'minimum_sentiment': np.min(sentiment_scores),
            'minimum_pos': np.argmin(sentiment_scores),
            'minimum_sentence': lines[np.argmin(sentiment_scores)],
            'maximum_sentiment': np.max(sentiment_scores),
            'maximum_pos': np.argmax(sentiment_scores),
            'maximum_sentence': lines[np.argmax(sentiment_scores)],
            'stdv_sentiment': np.std(sentiment_scores),
            'firstquartile_sentiment': np.percentile(sentiment_scores, 25),
            'median_sentiment': np.percentile(sentiment_scores, 50),
            'thirdquartile_sentiment': np.percentile(sentiment_scores, 75),
            'ratio_negative': count_negative / len(lines),
            'ratio_positive': count_positive / len(lines)
        }
    else:
        features = {
            'average_sentiment': np.nan,
            'minimum_sentiment': np.nan,
            'maximum_sentiment': np.nan,
            'stdv_sentiment': np.nan,
            'firstquartile_sentiment': np.nan,
            'median_sentiment': np.nan,
            'thirdquartile_sentiment': np.nan,
            'ratio_negative': np.nan,
            'ratio_positive': np.nan
        }

    # Return the requested feature
    return features.get(category, np.nan)
def wordcount(text):
    words = text.split()
    return len(words)

def lines(text):
    res = ""
    for i in text:
        if i.isupper():
            res += "*" + i
        else:
            res += i
    m = res.split("*")
    m.remove('')
    numlines = len(m)
    return numlines

def type_token_ratio(text):
    words = text.split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    return ttr

def ngrams(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    analyzer = vectorizer.build_analyzer()
    n_grams = analyzer(text)
    return len(n_grams)

def unique_ngrams(text, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    analyzer = vectorizer.build_analyzer()
    n_grams = analyzer(text)
    return len(set(n_grams))

