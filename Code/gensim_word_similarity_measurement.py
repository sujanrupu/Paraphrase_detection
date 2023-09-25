import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample text or sentence
text = "My name is Sangeet."

# Tokenize the text into words
words = word_tokenize(text)

# Train a Word2Vec model
model = Word2Vec([words], vector_size=100, window=5, min_count=1, sg=0)

# Replace 'word_to_find' with the word you want to find similar words for
word_to_find = "Sangeet"

# Find similar words
similar_words = model.wv.most_similar(word_to_find, topn=5)

# Print the similar words and their similarity scores
for word, score in similar_words:
    print(f"Similar word: {word}, Similarity score: {score}")

#Similar word: name, Similarity score: 0.06797593832015991
#Similar word: My, Similarity score: 0.004503030329942703
#Similar word: ., Similarity score: -0.010839177295565605
#Similar word: is, Similarity score: -0.023671656847000122