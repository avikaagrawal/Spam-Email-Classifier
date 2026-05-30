import streamlit as st
import joblib

model = joblib.load("model/spam_classifier.pkl")

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧"
)

st.title("📧 Spam Email Classifier")
st.write("Enter a message and detect whether it is spam.")

message = st.text_area("Message")

if st.button("Detect Spam"):

    prediction = model.predict([message])[0]

    if prediction == 1:
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")