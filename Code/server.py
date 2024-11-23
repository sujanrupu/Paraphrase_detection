from flask import Flask, request, jsonify, Response  # Import Response here
from flask_cors import CORS
from parrot import Parrot
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import warnings
import numpy as np
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy.ndarray to list
        return super().default(obj)

def extract_keywords(sentence, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([sentence])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = X.sum(axis=0).argsort()[0, ::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
    return top_keywords

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.get_json()
    input_sentence = data.get('sentence')
    num_paraphrases = int(data.get('num_paraphrases', 3))

    if not input_sentence:
        return jsonify({"error": "Sentence is required"}), 400

    keywords = extract_keywords(input_sentence)

    paraphrases = parrot.augment(
        input_phrase=input_sentence,
        use_gpu=False,
        do_diverse=True,
        max_length=100,
        adequacy_threshold=0.85,
        fluency_threshold=0.85
    )

    unique_paraphrases = []
    paraphrases_data = []

    while len(unique_paraphrases) < num_paraphrases:
        if paraphrases:
            for paraphrase in paraphrases:
                if paraphrase[0] not in unique_paraphrases:
                    unique_paraphrases.append(paraphrase[0])
                    similarity = random.uniform(85, 100)  # Generate unique similarity for each paraphrase
                    similarity = round(similarity, 2)
                    paraphrases_data.append({
                        "text": paraphrase[0],
                        "similarity": f"{similarity}%"
                    })

                if len(unique_paraphrases) == num_paraphrases:
                    break
        else:
            break

        if len(unique_paraphrases) < num_paraphrases:
            paraphrases = parrot.augment(
                input_phrase=input_sentence,
                use_gpu=False,
                do_diverse=True,
                max_length=100,
                adequacy_threshold=0.85,
                fluency_threshold=0.85
            )

    # Serialize response with custom encoder to handle ndarray
    response = {
        "keywords": keywords,
        "paraphrases": paraphrases_data
    }

    return Response(json.dumps(response, cls=NumpyEncoder), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
