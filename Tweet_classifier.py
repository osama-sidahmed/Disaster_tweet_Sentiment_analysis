# import streamlit
import streamlit as st

# the title
st.title("Disaster Tweet Classifier")

# import basic libraries
import numpy as np
import pandas as pd
import joblib

from tensorflow import keras

'''

#### This page classifies tweets based on whether the tweet corresponds to a real disaster or not. 
#### The classifier employes Deep Learning algorithms based on Natural Language Processing to classify tweets.

***

#### Enter the Tweet on the text box below then hit Enter:
'''


# load the saved model
@st.cache(allow_output_mutation=True)
def load_model(path, type):
    if type == 1: # preprocessing
        model = joblib.load(path)
    elif type == 2: # Keras model
         model = keras.models.load_model(path)
    return model

# Load model with the function.
preprocessing = load_model('data/pickle/preprocessing.joblib', 1)
model = load_model('data/pickle/model/', 2)
text = st.text_input('', 'This is the biggest disaster ever!')
processed_text = preprocessing.transform({text}).todense()
prediction = model.predict(processed_text)
if prediction < 0.4:
    'Are you kidding me, this is not a real disaster tweet!'
elif prediction >= 0.45 and prediction <= 0.55:
    'hmmm, I am not sure!'
else:
    'This sounds like a real disaster tweet!'


'''
***

Osama Sidahmed
'''
