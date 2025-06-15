import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import nltk
import re

from nltk.corpus import stopwords

# =================== NLTK DOWNLOAD ====================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# =================== TEXT CLEANING ====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([word for word in text.split() if word not in stop_words])

# ================ Load Model & Vectorizer ==============
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ================ Streamlit App Layout =================
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title("🧠 Emotion Detection from Text")

st.markdown("Enter your text below or upload a CSV to detect emotions.")

if 'results' not in st.session_state:
    st.session_state['results'] = []

# ================ User Text Input ======================
user_input = st.text_area("✍️ Enter a sentence here:")
if st.button("Detect Emotion"):
    if user_input.strip():
        clean_input = clean_text(user_input)
        vect_input = vectorizer.transform([clean_input])
        prediction = model.predict(vect_input)[0]

        emojis = {
            'joy': '😊', 'anger': '😠', 'sadness': '😢',
            'fear': '😨', 'love': '❤️', 'surprise': '😲'
        }
        colors = {
            'joy': 'green', 'anger': 'red', 'sadness': 'blue',
            'fear': 'purple', 'love': 'orange', 'surprise': 'pink'
        }

        emoji = emojis.get(prediction, '')
        color = colors.get(prediction, 'black')
        st.markdown(f"<h3 style='color:{color}'>Detected Emotion: {emoji} {prediction.upper()}</h3>", unsafe_allow_html=True)

        # Save to session state
        st.session_state['results'].append({
            'Text': user_input,
            'Cleaned Text': clean_input,
            'Predicted Emotion': prediction
        })
    else:
        st.warning("⚠️ Please enter some text.")

# ================ File Upload for Batch ================
uploaded_file = st.file_uploader("📂 Upload a CSV file (with 'text' column) for batch prediction", type=['csv'])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    if 'text' in batch_df.columns:
        batch_df['Cleaned Text'] = batch_df['text'].apply(clean_text)
        vect_batch = vectorizer.transform(batch_df['Cleaned Text'])
        batch_df['Predicted Emotion'] = model.predict(vect_batch)
        st.success("✅ Batch prediction completed.")
        st.dataframe(batch_df[['text', 'Predicted Emotion']])

        # Download button
        csv_download = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Predictions", csv_download, "batch_predictions.csv", "text/csv")
    else:
        st.error("❌ The uploaded CSV must contain a 'text' column.")

# ================ Show Real-Time Stats =================
if st.button("📊 Show Emotion Stats"):
    if st.session_state['results']:
        df = pd.DataFrame(st.session_state['results'])
        st.write("### 🧾 Prediction Table")
        st.dataframe(df)

        st.write("### 🥧 Real-Time Emotion Distribution")
        emotion_counts = df['Predicted Emotion'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("No predictions yet. Try entering some text!")

# ================ Export to CSV ========================
if st.button("📥 Download Previous Results"):
    if st.session_state['results']:
        results_df = pd.DataFrame(st.session_state['results'])
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
    else:
        st.info("No predictions to export yet.")
