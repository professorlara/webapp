#import os
import pandas as pd
import numpy as np
#import nltk
import re
#from textblob import TextBlob
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from predict_page import predict_arousal, load_model
from calculatefeatures import type_token_ratio,wordcount,lines,ngrams,unique_ngrams,wordclass


#import requirements.txt


st.markdown(
    """
    <style>
    .big-font {
        font-size:50px !important;
    }
    .medium-font {
        font-size:30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<p class="big-font">🎵Music emotion prediction🎵</p>', unsafe_allow_html=True)

st.markdown('<p class="medium-font">Please copy-paste the lyrics to your favourite song!😎</p>', unsafe_allow_html=True)


lyrics = st.text_area("Lyrics input✏️", height=100)

if st.button("Predict Emotion 🎤"):
    status_placeholder = st.empty()
    status_placeholder.write("Predicting emotion... 🔍")
    

    # Calculate required values
    num_words = wordcount(lyrics)
    number_lines = lines(lyrics)
    content_density = type_token_ratio(lyrics)
    num_5grams = ngrams(lyrics, 5)
    num_unique_5grams = unique_ngrams(lyrics, 5)
    num_4grams = ngrams(lyrics, 4)
    num_unique_4grams = unique_ngrams(lyrics, 4)
    num_trigrams = ngrams(lyrics, 3)
    num_unique_trigrams = unique_ngrams(lyrics, 3)
    num_bigrams = ngrams(lyrics, 2)
    num_unique_bigrams = unique_ngrams(lyrics, 2)

    adjective_count = wordclass(lyrics, 'adjective')
    noun_count = wordclass(lyrics, 'noun')
    base_verb_count = wordclass(lyrics, 'base_verb')
    preposition_count = wordclass(lyrics, 'preposition')
    personal_pronoun_count = wordclass(lyrics, 'personal_pronoun')
    non3rdpersonsingularpresent_verb_count = wordclass(lyrics, 'non3rdpersonsingularpresent_verb')
    thirdpersonsingularpresent_verb_count = wordclass(lyrics, '3rdpersonsingularpresent_verb')
    TOTAL_verb_count = wordclass(lyrics, 'total_verb')
    past_participle_verb_freq = wordclass(lyrics, 'past_participle_verb_freq')
    coordinating_conjunctions_freq = wordclass(lyrics, 'coordinating_conjunctions_freq')

    # Initialize dictionary
    result_dict = {
        "num_words": num_words,
        "number_lines": number_lines,
        "content_density": content_density,
        "num_5grams": num_5grams,
        "num_unique_5grams": num_unique_5grams,
        "num_4grams": num_4grams,
        "num_unique_4grams": num_unique_4grams,
        "num_trigrams": num_trigrams,
        "num_unique_trigrams": num_unique_trigrams,
        "num_bigrams": num_bigrams,
        "num_unique_bigrams": num_unique_bigrams,
        "adjective_count": adjective_count,
        "noun_count": noun_count,
        "base_verb_count": base_verb_count,
        "preposition_count": preposition_count,
        "personal_pronoun_count": personal_pronoun_count,
        "non3rdpersonsingularpresent_verb_count": non3rdpersonsingularpresent_verb_count,
        "thirdpersonsingularpresent_verb_count": thirdpersonsingularpresent_verb_count,
        "TOTAL_verb_count": TOTAL_verb_count,
        "past_participle_verb_freq": past_participle_verb_freq,
        "coordinating_conjunctions_freq": coordinating_conjunctions_freq
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([result_dict])

    #st.write(df)


    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    features = [
        'adjective_count', 'num_words', 'num_5grams', 'num_4grams',
        'num_unique_bigrams', 'num_unique_trigrams', 'num_unique_5grams',
        'num_trigrams', 'number_lines', 'num_bigrams',
        'non3rdpersonsingularpresent_verb_count', 'num_unique_4grams',
        'noun_count', 'base_verb_count', 'TOTAL_verb_count', 'content_density',
        'personal_pronoun_count','coordinating_conjunctions_freq',
        'preposition_count'
        ]

    # Load the training and validation data
    train = pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/TRAIN language data.csv',delimiter=";")
    test = pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/TEST language data.csv',delimiter=";")
    validate= pd.read_csv('C:/Users/larae/OneDrive - Tromso kommune/Skrivebord/webapp/VAL language data.csv',delimiter=";")


    # Combine the train and validate data
    train_validate = pd.concat([train, validate], ignore_index=True, sort=False)

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
    #load_model()


    #new_predictions = predict_arousal(df)

    df = df[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    df = df.replace((np.inf, -np.inf, np.nan), 0)
    new_predictionsA = model.predict(df)

    
    status_placeholder.write("This may take some time... ⌛")





    #Valence
    # Extract the relevant features and target variable
    X = train_validate[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    y = train_validate['valence_tags'].apply(lambda x: str(x).replace(',', '.')).astype(float)

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
    #load_model()


    #new_predictions = predict_arousal(df)

    df = df[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    df = df.replace((np.inf, -np.inf, np.nan), 0)
    new_predictionsV = model.predict(df)

    


    #Dominance
    # Extract the relevant features and target variable
    X = train_validate[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    y = train_validate['dominance_tags'].apply(lambda x: str(x).replace(',', '.')).astype(float)

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
    #load_model()


    #new_predictions = predict_arousal(df)

    df = df[features].applymap(lambda x: str(x).replace(',', '.')).astype(float)
    df = df.replace((np.inf, -np.inf, np.nan), 0)
    new_predictionsD = model.predict(df)
    
    status_placeholder.empty()
    st.write("Prediction completed! 🎉")
    
    #st.write("Arousal:", round(new_predictionsA[0],2))
    #st.write("Valence:", round(new_predictionsV[0],2))
    #st.write("Dominance:", round(new_predictionsD[0],2))

    percentageA =  round((new_predictionsA[0]/7)*100,1)

    

    #TEXT TO GO WITH PLOT

    if new_predictionsA < 4:
        #st.write("This song has a low arousal rating of",str(percentageA), "%.")
        st.write("This arousal rating suggests that it is calm and relaxing.")
        colourA = 'lightblue'
    elif new_predictionsA >= 4 and new_predictionsA <=5:
        #st.write("This song has a moderate arousal rating of", str(percentageA), "%.")
        st.write("This arousal rating suggests that it is upbeat and rhythimical.")
        colourA = 'lightgreen'
    else:
        #st.write("This song has a high arousal rating of", str(percentageA), "%.")
        st.write("This arousal rating suggests that it is exciting and energetic.")
        colourA = 'orange'

    #PLOT

    import matplotlib.pyplot as plt


    arousal_rating = percentageA


    fig, ax = plt.subplots(figsize=(10, 2))


    sections = ['Low', 'Moderate', 'High']
    colors = ['lightblue', 'lightgreen', 'orange']
    positions = [0, 33.33, 66.66, 100]  #


    for i in range(len(sections)):
        ax.barh(0, positions[i + 1] - positions[i], left=positions[i], color=colors[i], edgecolor='white', height=1.0)

    #Needle
    needle_position = arousal_rating
    ax.plot([needle_position, needle_position], [-0.5, 0.5], color='black', linewidth=2)
    ax.text(needle_position, 0.65, f'{arousal_rating}%', horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

    #Labels
    label_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(len(sections))]
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(sections)


    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title('Arousal Rating',y=-0.4)
    st.pyplot(fig)

    
    #TEXT TO GO WITH PLOT
    percentageV =  round((new_predictionsV[0]/7)*100,1)

    if new_predictionsV < 4:
        #st.write("This song has a low valence rating of",str(percentageV), "%.")
        st.write("This valence rating suggests that it is melancholic and sad.")
        colourV = 'lightblue'
    elif new_predictionsV >= 4 and new_predictionsV <=5:
        #st.write("This song has a moderate valence rating of", str(percentageV), "%.")
        st.write("This valence rating suggests that it is pleasant and neutral.")
        colourV = 'lightgreen'
    else:
        #st.write("This song has a high valence rating of", str(percentageV), "%.")
        st.write("This valence rating suggests that it is joyful and happy.")
        colourV = 'orange'
    #PLOT

    valence_rating = percentageV


    fig, ax = plt.subplots(figsize=(10, 2))


    sections = ['Low', 'Moderate', 'High']
    colors = ['lightblue', 'lightgreen', 'orange']
    positions = [0, 33.33, 66.66, 100]  


    for i in range(len(sections)):
        ax.barh(0, positions[i + 1] - positions[i], left=positions[i], color=colors[i], edgecolor='white', height=1.0)

    #Needle
    needle_position = valence_rating
    ax.plot([needle_position, needle_position], [-0.5, 0.5], color='black', linewidth=2)
    ax.text(needle_position, 0.65, f'{valence_rating}%', horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

    #Labels
    label_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(len(sections))]
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(sections)


    for spine in ax.spines.values():
        spine.set_visible(False)
    

    plt.title('Valence Rating',y=-0.4)
    st.pyplot(fig)

     #TEXT TO GO WITH PLOT

    percentageD =  round((new_predictionsD[0]/7)*100,1)


    if new_predictionsD < 4:
        #st.write("This song has a low dominance rating of",str(percentageD), "%.")
        st.write("This dominance rating suggests that it is gentle and reserved.")
        colourD = 'lightblue'
    elif new_predictionsD >= 4 and new_predictionsD <=5:
        #st.write("This song has a moderate dominance rating of", str(percentageD), "%.")
        st.write("This dominance rating suggests that it is impactful and controlled.")
        colourD = 'lightgreen'
    else:
        #st.write("This song has a high dominance rating of", str(percentageD), "%.")
        st.write("This dominance rating suggests that it is powerful and commanding.")
        colourD = 'orange'
    #PLOT

    dominance_rating = percentageD


    fig, ax = plt.subplots(figsize=(10, 2))


    sections = ['Low', 'Moderate', 'High']
    colors = ['lightblue', 'lightgreen', 'orange']
    positions = [0, 33.33, 66.66, 100]  #


    for i in range(len(sections)):
        ax.barh(0, positions[i + 1] - positions[i], left=positions[i], color=colors[i], edgecolor='white', height=1.0)

    #Needle
    needle_position = dominance_rating
    ax.plot([needle_position, needle_position], [-0.5, 0.5], color='black', linewidth=2)
    ax.text(needle_position, 0.65, f'{dominance_rating}%', horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

    #Labels
    label_positions = [(positions[i] + positions[i + 1]) / 2 for i in range(len(sections))]
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(sections)


    for spine in ax.spines.values():
        spine.set_visible(False)
    

    plt.title('Dominance Rating',y=-0.4)
    st.pyplot(fig)


    
    

    
    


        
    


    



