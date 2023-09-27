import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
nltk.download('punkt')

# Sample sentences
sentences = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Word embeddings are used in various NLP tasks.",
    "Machine learning algorithms can be applied to text data.",
    "The cat is on the mat.",
    "A dog is chasing a ball in the park."
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Function to calculate sentence vectors
def sentence_vector(sentence, model):
    words = word_tokenize(sentence.lower())
    vector = np.zeros(model.vector_size)
    word_count = 0
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
            word_count += 1
    if word_count > 0:
        vector /= word_count
    return vector

# Calculate sentence vectors
sentence_vectors = [sentence_vector(sentence, model) for sentence in sentences]

# Calculate cosine similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(sentence_vectors, sentence_vectors)

# Print the similarity matrix
for i in range(len(sentences)):
    for j in range(len(sentences)):
        print(f"Similarity between Sentence {i+1} and Sentence {j+1}: {similarities[i][j]}")




# Output:
#Similarity between Sentence 1 and Sentence 1: 1.0000000000000002
#Similarity between Sentence 1 and Sentence 2: 0.32410557530638356
#Similarity between Sentence 1 and Sentence 3: 0.10653615303211826
#Similarity between Sentence 1 and Sentence 4: 0.2549468749113887 
#Similarity between Sentence 1 and Sentence 5: 0.36875390199142066
#Similarity between Sentence 2 and Sentence 1: 0.32410557530638356
#Similarity between Sentence 2 and Sentence 2: 1.0000000000000002
#Similarity between Sentence 2 and Sentence 3: 0.15402169916901404
#Similarity between Sentence 2 and Sentence 4: 0.10977352233949632
#Similarity between Sentence 2 and Sentence 5: 0.33400411489426723
#Similarity between Sentence 3 and Sentence 1: 0.10653615303211826
#Similarity between Sentence 3 and Sentence 2: 0.15402169916901404
#Similarity between Sentence 3 and Sentence 3: 0.9999999999999998
#Similarity between Sentence 3 and Sentence 4: 0.12575358900221129
#Similarity between Sentence 3 and Sentence 5: -0.004833370785682675
#Similarity between Sentence 4 and Sentence 1: 0.2549468749113887
#Similarity between Sentence 4 and Sentence 2: 0.10977352233949632
#Similarity between Sentence 4 and Sentence 3: 0.12575358900221129
#Similarity between Sentence 4 and Sentence 4: 1.0000000000000004
#Similarity between Sentence 4 and Sentence 5: 0.24471492254355748
#Similarity between Sentence 5 and Sentence 1: 0.36875390199142066
#Similarity between Sentence 5 and Sentence 2: 0.3340041148942672
#Similarity between Sentence 5 and Sentence 3: -0.0048333707856826474
#Similarity between Sentence 5 and Sentence 4: 0.2447149225435575
#Similarity between Sentence 5 and Sentence 5: 1.0000000000000004
