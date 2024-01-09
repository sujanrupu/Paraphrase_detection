import re
import requests
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import tensorflow_hub as hub

def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)

import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords, punctuation, and lemmatize the words
    stop_words = set(stopwords.words('english'))
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

    return ' '.join(words)

def nltk_similarity(text1, text2):
  # Load the Universal Sentence Encoder (USE)
  use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  # Load the two text files
  text1 = 'data1.txt'
  text2 = 'data2.txt'

  with open(text1, 'r') as file1, open(text2, 'r') as file2:
      lines_file1 = file1.readlines()[1:10000]
      lines_file2 = file2.readlines()[1:10000]

  # Preprocess the lines from both files
  lines_file1 = [preprocess_text(line) for line in lines_file1]
  lines_file2 = [preprocess_text(line) for line in lines_file2]

  # Calculate the average cosine similarity between lines using USE embeddings
  average_similarity = 0

  for line1, line2 in zip(lines_file1, lines_file2):
      embeddings1 = use_embed([line1])
      embeddings2 = use_embed([line2])

      similarity = np.inner(embeddings1, embeddings2)
      average_similarity += similarity[0][0]

  average_similarity /= len(lines_file1)
  return average_similarity

import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

def calculate_semantic_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Make API request to ConceptNet
    url = f'http://api.conceptnet.io/relatedness?node1=/c/en/{text1}&node2=/c/en/{text2}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        similarity = data['value']
        return similarity
    else:
        print(f"API request failed with status code {response.status_code}")
        return 0.0

  def calculate_accuracy_range(file1_lines, file2_lines, start, end):
    similarity_scores = []

    for i in range(start, end):
        line1 = file1_lines[i]
        line2 = file2_lines[i]

        similarity = calculate_semantic_similarity(line1, line2)
        similarity_scores.append(similarity)

    return similarity_scores

def conceptnet_similarity(text1, text2):
  # Read lines from both text files
  with open('data1.txt', 'r', encoding='utf-8') as file1, open('data2.txt', 'r', encoding='utf-8') as file2:
      text1 = file1.readlines()
      text2 = file2.readlines()

  # Define the range of lines to process
  start_line = 0
  end_line = 600

  # Set the number of concurrent threads (adjust as needed)
  num_threads = 4

  # Split the range into chunks for parallel processing
  chunk_size = (end_line - start_line) // num_threads
  chunks = [(i, min(i + chunk_size, end_line)) for i in range(start_line, end_line, chunk_size)]

  # Create a ThreadPoolExecutor for concurrent processing
  with ThreadPoolExecutor(max_workers=num_threads) as executor:
      futures = [executor.submit(calculate_accuracy_range, text1, text2, start, end) for start, end in chunks]

      # Gather results from completed tasks
      similarity_scores = []
      for future in as_completed(futures):
          similarity_scores.extend(future.result())

  # Calculate the average similarity score
  average_similarity = sum(similarity_scores) / len(similarity_scores)

  print(f'Semantic Similarity Accuracy: {average_similarity}')

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords, punctuation, and lemmatize the words
    stop_words = set(stopwords.words('english'))
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

    return ' '.join(words)

def bert_similarity(text1, text2):
  # Load a pretrained BERT-based model
  model_name = "bert-base-nli-stsb-mean-tokens"
  model = SentenceTransformer(model_name)

  # Load the two text files
  text1 = 'data1.txt'
  text2 = 'data2.txt'

  with open(text1, 'r') as file1, open(text2, 'r') as file2:
      lines_file1 = file1.readlines()[2:600]
      lines_file2 = file2.readlines()[2:600]

  # Preprocess the lines from both files
  lines_file1 = [preprocess_text(line) for line in lines_file1]
  lines_file2 = [preprocess_text(line) for line in lines_file2]

  # Calculate the average cosine similarity between lines using BERT embeddings
  average_similarity = 0

  for line1, line2 in zip(lines_file1, lines_file2):
      embeddings1 = model.encode(line1, convert_to_tensor=True)
      embeddings2 = model.encode(line2, convert_to_tensor=True)

      similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
      average_similarity += similarity.item()

  average_similarity /= len(lines_file1)
  return average_similarity

def base_model_predictions(text1, text2):
    nltk_sim = nltk_similarity(text1, text2)
    conceptnet_sim = conceptnet_similarity(text1, text2)
    bert_sim = bert_similarity(text1, text2)
    return [nltk_sim, conceptnet_sim, bert_sim]

# Ensemble model prediction
def ensemble_prediction(text1, text2, base_models):
    return np.mean(base_models)  # Simple averaging as a demonstration

# Load text files
with open('data1.txt', 'r', encoding='utf-8') as file1, open('data2.txt', 'r', encoding='utf-8') as file2:
    lines_file1 = file1.readlines()[:5]  # Adjust the range as needed
    lines_file2 = file2.readlines()[:5]  # Adjust the range as needed

preprocessed_lines = [(preprocess_text(line1), preprocess_text(line2)) for line1, line2 in zip(lines_file1, lines_file2)]
# Function to preprocess text
def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)

# Load text files
with open('data1.txt', 'r', encoding='utf-8') as file1, open('data2.txt', 'r', encoding='utf-8') as file2:
    lines_file1 = file1.readlines()[:600]  # Adjust the range as needed
    lines_file2 = file2.readlines()[:600]  # Adjust the range as needed

# Preprocess all lines beforehand
preprocessed_lines = [(preprocess_text(line1), preprocess_text(line2)) for line1, line2 in zip(lines_file1, lines_file2)]

# Base model predictions for all pairs of preprocessed lines
base_model_predictions_list = []

for line1, line2 in preprocessed_lines:
    # Perform model predictions directly or call respective functions here
    # Replace these placeholder functions with actual implementation
    nltk_sim = nltk_similarity(line1, line2)
    conceptnet_sim = conceptnet_similarity(line1, line2)
    bert_sim = bert_similarity(line1, line2)

    predictions = [nltk_sim, conceptnet_sim, bert_sim]
    base_model_predictions_list.append(predictions)

# Train the ensemble model on base model predictions
ensemble_predictions = []

for predictions in base_model_predictions_list:
    # Perform ensemble prediction or call respective function here
    ensemble_pred = ensemble_prediction(line1, line2, predictions)
    ensemble_predictions.append(ensemble_pred)

# Calculate the average ensemble prediction score
average_ensemble_prediction = np.mean(ensemble_predictions)

print("Average Ensemble Prediction:", average_ensemble_prediction)

# Average Ensemble Prediction: 90.34
