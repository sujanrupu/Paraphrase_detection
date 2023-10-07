import gensim.downloader as api
import re
import numpy as np

# Load a pre-trained Word2Vec model (e.g., 'word2vec-google-news-300')
word2vec_model = api.load("word2vec-google-news-300")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

# Function to calculate cosine similarity between two sentences using Word2Vec
def calculate_cosine_similarity(sentence1, sentence2, model):
    tokens1 = preprocess_text(sentence1).split()
    tokens2 = preprocess_text(sentence2).split()

    # Filter tokens that are not in the Word2Vec model's vocabulary
    tokens1 = [token for token in tokens1 if token in model]
    tokens2 = [token for token in tokens2 if token in model]

    if not tokens1 or not tokens2:
        return 0.0

    vec1 = np.mean([model[token] for token in tokens1], axis=0)
    vec2 = np.mean([model[token] for token in tokens2], axis=0)

    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return similarity

# Function to calculate accuracy
def calculate_accuracy(file1_lines, file2_lines, word2vec_model):
    total_lines = min(len(file1_lines), len(file2_lines))
    similarity_scores = []

    for i in range(2, 1000):
        line1 = file1_lines[i]
        line2 = file2_lines[i]

        similarity = calculate_cosine_similarity(line1, line2, word2vec_model)
        similarity_scores.append(similarity)

    # Calculate the average similarity score
    average_similarity = sum(similarity_scores) / len(similarity_scores)

    return average_similarity

# Read lines from both text files
with open('data1.txt', 'r', encoding='utf-8') as file1, open('data2.txt', 'r', encoding='utf-8') as file2:
    file1_lines = file1.readlines()
    file2_lines = file2.readlines()

# Calculate accuracy
accuracy = calculate_accuracy(file1_lines, file2_lines, word2vec_model)

print(f'Semantic Similarity Accuracy: {accuracy}')


# Semantic Similarity Accuracy: 98.21716308713199%
