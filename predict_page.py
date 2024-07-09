import os
import streamlit as st
import pandas as pd
import numpy as np
import spacy

import nltk

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error




def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # Load the training and validation data
    train = pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/TRAIN language data.csv',delimiter=";")
    test = pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/TEST language data.csv',delimiter=";")
    validate= pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/VAL language data.csv',delimiter=";")


    # Combine the train and validate data
    train_validate = pd.concat([train, validate], ignore_index=True, sort=False)

    # Define the relevant features
    features = [
        'adjective_count', 'num_words', 'num_5grams', 'num_4grams',
        'num_unique_bigrams', 'num_unique_trigrams', 'num_unique_5grams',
        'num_trigrams', 'number_lines', 'num_bigrams',
        'non3rdpersonsingularpresent_verb_count', 'num_unique_4grams',
        'noun_count', 'base_verb_count', 'TOTAL_verb_count', 'content_density',
        'personal_pronoun_count','coordinating_conjunctions_freq',
        'preposition_count'
    ]

    # Extract the relevant features and target variable
    X = train_validate[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    y = train_validate['arousal_tags'].apply(lambda x: str(x).replace(',', '.')).astype(float)

    # Replace inf, -inf, and NaN with 0
    X = X.replace((np.inf, -np.inf, np.nan), 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the GradientBoostingRegressor model
    model = GradientBoostingRegressor(
        criterion='squared_error',
        learning_rate=0.1,
        loss='absolute_error',
        min_samples_leaf=10,
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Function to predict arousal on new data
def predict_arousal(new_data):
    features = [
    'adjective_count', 'num_words', 'num_5grams', 'num_4grams',
    'num_unique_bigrams', 'num_unique_trigrams', 'num_unique_5grams',
    'num_trigrams', 'number_lines', 'num_bigrams',
    'non3rdpersonsingularpresent_verb_count', 'num_unique_4grams',
    'noun_count', 'base_verb_count', 'TOTAL_verb_count', 'content_density',
    'personal_pronoun_count','coordinating_conjunctions_freq',
    'preposition_count'
    ]

    new_data = new_data[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    new_data = new_data.replace((np.inf, -np.inf, np.nan), 0)
    prediction = model.predict(new_data)
    return prediction
