from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import spacy
from unidecode import unidecode
import docx2txt
from pypdf import PdfReader 

crf = joblib.load('crf_model.joblib')

nlp = spacy.load('fr_core_news_sm')

def preprocess_phrase(phrase):
    phrase_ascii = unidecode(phrase)  
    doc = nlp(phrase_ascii)
    words = [token.text for token in doc]
    return words

def word2features(sent, i):
    word = sent[i]
    features = {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sent[i - 1],
        'next_word': '' if i == len(sent) - 1 else sent[i + 1],
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def extract_text_from_docx(file):
    return docx2txt.process(file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/predict', methods=['POST'])
@cross_origin(origins="http://localhost:3000")
def predict():
    data = request.json
    phrase = data['phrase']
    
    words = preprocess_phrase(phrase)
    features = sent2features(words)
    
    predictions = crf.predict([features])[0]
    
    result = {"phrase": phrase, "annotations": [{"mot": word, "annotation": annotation} for word, annotation in zip(words, predictions)]}
    
    print(result)
    
    return jsonify(result)

@app.route('/upload', methods=['POST'])
@cross_origin(origins="http://localhost:3000")
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        words = preprocess_phrase(text)
        features = sent2features(words)
        predictions = crf.predict([features])[0]
    
        result = {"phrase": text, "annotations": [{"mot": word, "annotation": annotation} for word, annotation in zip(words, predictions)]}
        
        return jsonify({result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
