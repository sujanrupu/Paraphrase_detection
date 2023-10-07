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

# Load a pretrained BERT-based model
model_name = "bert-base-nli-stsb-mean-tokens"
model = SentenceTransformer(model_name)

# Load the two text files
file1_path = 'data1.txt'
file2_path = 'data2.txt'

with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
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

print("Average Cosine Similarity (using BERT):", average_similarity)


# Average Cosine Similarity (using BERT): 97.1153025344064%
