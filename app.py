import streamlit as st

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

    status_placeholder.write("This may take some time... ⌛")

    status_placeholder.empty()
    st.write("Prediction completed! 🎉")
