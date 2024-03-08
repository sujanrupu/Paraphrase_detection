import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords, punctuation, and lemmatize the words
    stop_words = set(stopwords.words('english'))
    words = [WordNetLemmatizer().lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    
    return ' '.join(words)

# Load the Universal Sentence Encoder (USE)
use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load the two text files
file1_path = 'data1.txt'
file2_path = 'data2.txt'

with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
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

print("Average Cosine Similarity (using USE):", average_similarity)




# Average similairty: 86.4119%
