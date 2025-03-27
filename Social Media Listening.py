import streamlit as st
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load saved model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer (3).pkl')
svm_model = joblib.load('emotion_classification_model.pkl')

def predict_emotion(text):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = svm_model.predict(transformed_text)
    emotions = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    return emotions.get(prediction[0], 'Unknown')

# Streamlit UI
st.title("Emotion Detection App")
st.write("Enter a text message to predict its emotion.")

user_input = st.text_area("Enter your message here:")
if st.button("Predict Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: {emotion}")
    else:
        st.warning("Please enter some text before predicting.")
