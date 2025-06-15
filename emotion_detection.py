# ============================== IMPORTS ==============================
import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ========================== NLTK DOWNLOADS ===========================
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ========================== TEXT CLEANING ===========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\W', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# =========================== LOAD DATA ==============================
def load_data(filename):
    texts, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if ';' in line:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    texts.append(clean_text(parts[0]))  # clean text here
                    labels.append(parts[1])
    return pd.DataFrame({'text': texts, 'label': labels})

# ✅ Load and preprocess data
train_df = load_data('train.txt')
test_df = load_data('test.txt')

print("\n📦 Sample training data:")
print(train_df.head())

# ==================== FEATURE EXTRACTION (TF-IDF) ===================
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']

# ========================== MODEL TRAINING ==========================
print("\n🧠 Training Multinomial Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# ========================== EVALUATION ==============================
y_pred_nb = nb_model.predict(X_test)
print("\n✅ Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\n📊 Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# ======================== EMOTION DISTRIBUTION ======================
plt.figure(figsize=(8,5))
sns.countplot(x='label', data=train_df, order=train_df['label'].value_counts().index)
plt.title("Emotion Distribution in Training Data")
plt.xticks(rotation=45)
plt.tight_layout()
cm = confusion_matrix(y_test, y_pred_nb, labels=nb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ================ LOGISTIC REGRESSION & SVM =========================
# ▶ Logistic Regression
log_model = LogisticRegression(max_iter=1000,class_weight='balanced')
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("\n🧠 Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))

# ▶ SVM (Support Vector Machine)
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("\n🧠 SVM Accuracy:", accuracy_score(y_test, svm_pred))

# ===================== SAVE BEST MODEL & VECTORIZER =================
joblib.dump(nb_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(svm_model, "model.pkl")
print("\n💾 Model and vectorizer saved successfully.")





report = classification_report(y_test, svm_pred, output_dict=True)
for emotion, scores in report.items():
    if emotion in ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']:
        print(f"{emotion.capitalize():<10} - Precision: {scores['precision']:.2f}, Recall: {scores['recall']:.2f}, F1-Score: {scores['f1-score']:.2f}")

print("\n📊 Model Accuracies:")
print(f"Naive Bayes:        {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Logistic Regression:{accuracy_score(y_test, log_pred):.4f}")
print(f"SVM:                {accuracy_score(y_test, svm_pred):.4f}")

