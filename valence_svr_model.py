

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import joblib

# Load the training and validation data
train = pd.read_csv('C:/Users/Lara/Desktop/webapp/TRAIN language data.csv', delimiter=";")
validate = pd.read_csv('C:/Users/Lara/Desktop/webapp/VAL language data.csv', delimiter=";")

# Combine the train and validate data
train_validate = pd.concat([train, validate], ignore_index=True, sort=False)

# Define the features
features = [
        'num_words','number_lines','content_density', 'num_5grams',
        'num_unique_5grams','num_4grams','num_unique_4grams','num_trigrams',
        'num_unique_trigrams', 'num_bigrams',
        'num_unique_bigrams','adjective_count','noun_count',
        'base_verb_count','preposition_count',
        'personal_pronoun_count', 'non3rdpersonsingularpresent_verb_count',
        'TOTAL_verb_count',
        'coordinating_conjunctions_freq','HF_byline_average_sentiment',
        'HF_byline_minimum_sentiment',
        'HF_byline_maximum_sentiment','HF_byline_stdv_sentiment',
        'HF_byline_firstquartile_sentiment', 'HF_byline_median_sentiment',
        'HF_byline_thirdquartile_sentiment', 'HF_byline_ratio_negative',
        'HF_byline_ratio_positive', #'VADER_byline_average_sentiment'
        ]

# Ensure all required columns are present
missing_columns = set(features) - set(train_validate.columns)
for col in missing_columns:
    train_validate[col] = np.nan

# Extract the relevant features and target variable
X = train_validate[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
y = train_validate['valence_tags'].apply(lambda x: str(x).replace(',', '.')).astype(float)

# Replace inf, -inf, and NaN with 0
X = X.replace((np.inf, -np.inf, np.nan), 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVR model
model = SVR(
    kernel='linear',
    epsilon=0.5,
    degree=1,
    C=0.05
)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'svr_model.pkl')