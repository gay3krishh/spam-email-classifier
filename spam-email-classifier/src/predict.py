import joblib
import sys

# Load model and vectorizer
model = joblib.load("model/spam_classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_message(message):
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]
    return "SPAM" if prediction == 1 else "HAM"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py 'Your message here'")
    else:
        message = " ".join(sys.argv[1:])
        result = predict_message(message)
        print(f"\nMessage: {message}\nPrediction: {result}")
