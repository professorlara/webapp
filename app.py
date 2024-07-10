import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from calculatefeatures import type_token_ratio,wordcount,lines,ngrams,unique_ngrams

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

    #This is where the prediction for arousal goes


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
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([result_dict])
    
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    features = [
        'num_words', 'num_5grams', 'num_4grams',
        'num_unique_bigrams', 'num_unique_trigrams', 'num_unique_5grams',
        'num_trigrams', 'number_lines', 'num_bigrams',
        'num_unique_4grams', 'content_density'
        ]
    train = pd.read_csv('TRAIN language data_1.csv',delimiter=";")
    test = pd.read_csv('TEST language data_1.csv',delimiter=";")
    validate= pd.read_csv('VAL language data_1.csv',delimiter=";")

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

    

    #This is where the prediction for valence/donimance goes

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


    percentageA =  round((new_predictionsA[0]/7)*100,1)
   

    #TEXT TO GO WITH PLOT




    if new_predictionsA < 4:
        #st.write("This song has a low arousal rating of",str(percentageA), "%.")
        st.markdown('''This :blue[low] arousal rating suggests that the song is calm and relaxing.''')
        colourA = 'lightblue'
    elif new_predictionsA >= 4 and new_predictionsA <=5:
        #st.write("This song has a moderate arousal rating of", str(percentageA), "%.")
        st.markdown('''
        This :green[moderate] arousal rating suggests that the song is upbeat and rhythimical.''')
        colourA = 'lightgreen'
    else:
        #st.write("This song has a high arousal rating of", str(percentageA), "%.")
        st.markdown('''
        This :orange[high] arousal rating suggests that the song is exciting and energetic.")
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
        st.markdown('''This :blue[low] valence rating suggests that the song is melancholic and sad.''')
        colourV = 'lightblue'
    elif new_predictionsV >= 4 and new_predictionsV <=5:
        #st.write("This song has a moderate valence rating of", str(percentageV), "%.")
        st.markdown('''This :green[moderate] valence rating suggests that the song is pleasant and neutral.''')
        colourV = 'lightgreen'
    else:
        #st.write("This song has a high valence rating of", str(percentageV), "%.")
        st.markdown('''This :orange[high] valence rating suggests that the song is joyful and happy.''')
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
        st.markdown('''This :blue[low] dominance rating suggests that the song is gentle and reserved.''')
        colourD = 'lightblue'
    elif new_predictionsD >= 4 and new_predictionsD <=5:
        #st.write("This song has a moderate dominance rating of", str(percentageD), "%.")
        st.markdown('''This :green[moderate] dominance rating suggests that the song is impactful and controlled.''')
        colourD = 'lightgreen'
    else:
        #st.write("This song has a high dominance rating of", str(percentageD), "%.")
        st.markdown('''
        This :orange[high] dominance rating suggests that the song is impactful and controlled.''')        
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



    import pygal
    from pygal.style import Style

    custom_style = Style(
        value_font_size=30,  # Adjust font size here
        value_colors=('black',),  
        label_font_size=0,
        value_label_font_size = 20
    )
    
    def create_gauge_chart(title, percentage, color):
        gauge = pygal.SolidGauge(
            half_pie=True,
            inner_radius=0.70,
            show_legend=False,
            style=custom_style
        )
        gauge.title = title  # Set title for the gauge
        gauge.add('', [{'value': percentage, 'max_value': 100, 'color': color}])
        return gauge.render(is_unicode=True)
        
    # Example data for three gauge charts
    titles = ['Arousal', 'Valence', 'Dominance']
    percentages = [percentageA, percentageV, percentageD]
    colors = [colourA, colourV, colourD]
    
    gauge_svgs = []
    for title, percentage, color in zip(titles, percentages, colors):
        gauge_svgs.append(create_gauge_chart(title, percentage, color))
    
    # Display gauge charts side by side in Streamlit
    #st.write("<div style='display:flex;'>")
    for gauge_svg in gauge_svgs:
        st.write(f"<div style='margin: auto;'>{gauge_svg}</div>", unsafe_allow_html=True)
    #st.write("</div>")
        
    

            
