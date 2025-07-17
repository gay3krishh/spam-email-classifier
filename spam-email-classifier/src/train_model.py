import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

nltk.download('punkt')

# Updated dataset path
data_path = "data/spam.csv"
df = pd.read_csv(data_path, encoding="latin-1")

# Some versions of this dataset may contain unnamed columns. Drop them if needed.
df = df[['v1', 'v2']]  # v1 = label, v2 = text
df.columns = ['label', 'text']  # rename

# Convert labels to binary: 'spam' = 1, 'ham' = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Text vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
