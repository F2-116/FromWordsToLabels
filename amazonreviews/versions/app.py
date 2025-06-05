from flask import Flask, render_template, request, jsonify
import re
import contractions
import inflect
import spacy
from nltk.corpus import stopwords
import langid
import joblib
from scipy.sparse import hstack
import numpy as np

app = Flask(__name__)

# Initialize resources
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "n't"}
inflect_engine = inflect.engine()

# Load models
tfidf = joblib.load("train.ft.txt/tfidf_vectorizer.joblib")
model = joblib.load("linear_svc_l2_model.joblib")

def detect_language(text: str) -> str:
    try:
        if isinstance(text, str) and text.strip():
            return langid.classify(text)[0]
    except Exception:
        pass
    return "unknown"

def remove_punctuation_except(text: str, keep: str = "!?") -> str:
    return "".join(char for char in text if char.isalnum() or char.isspace() or char in keep)

def convert_numbers_to_words(text: str) -> str:
    def replacer(match):
        num_str = match.group()
        try:
            if num_str.endswith('.') and num_str.count('.') == 1:
                num_str = num_str.rstrip('.')
            if '.' in num_str:
                int_part, dec_part = num_str.split('.', 1)
                int_words = inflect_engine.number_to_words(int_part, andword='') if int_part else ''
                dec_words = ' '.join(inflect_engine.number_to_words(d) for d in dec_part if d.isdigit()) if dec_part else ''
                return f"{int_words} point {dec_words}".strip()
            return inflect_engine.number_to_words(num_str, andword='')
        except Exception:
            return num_str
    return re.sub(r'\d+(\.\d+)?', replacer, text)

def preprocess_raw_text(text: str) -> str:
    text = contractions.fix(text)
    text = text.strip().lower()
    text = remove_punctuation_except(text, keep="!?")
    text = convert_numbers_to_words(text)
    return " ".join(text.split())

def preprocess_and_lemmatize(text: str) -> str:
    cleaned_text = preprocess_raw_text(text)
    doc = nlp(cleaned_text)
    lemmas = [
        token.lemma_ for token in doc
        if token.text not in stop_words and not token.is_punct and not token.is_space
    ]
    return " ".join(lemmas)

def add_text_features(text: str) -> np.ndarray:
    failed_number_conversion = 0
    sentence_length = len(text.split())
    num_exclamations = text.count('!') + text.count('?')
    return np.array([[failed_number_conversion, sentence_length, num_exclamations]], dtype=float)

def is_valid_english_sentence(text: str) -> bool:
    if not text or not any(c.isalpha() for c in text):
        return False
    return detect_language(text) == 'en'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("review", "")
    
    if not is_valid_english_sentence(user_input):
        return jsonify({"result": "Please enter a valid English sentence."})

    processed_input = preprocess_and_lemmatize(user_input)
    tfidf_vector = tfidf.transform([processed_input])
    numeric_features = add_text_features(processed_input)
    combined_features = hstack([tfidf_vector, numeric_features])
    prediction = model.predict(combined_features)

    result_text = "This is a negative rating." if prediction[0] == 1 else "This is a positive rating."
    return jsonify({"result": result_text})

if __name__ == "__main__":
    app.run(debug=True)
