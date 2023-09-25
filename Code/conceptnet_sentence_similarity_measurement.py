import requests
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

def get_word_similarity(word1, word2):
    # Define the ConceptNet API endpoint for similarity
    api_url = f"http://api.conceptnet.io/relatedness?node1=/c/en/{word1}&node2=/c/en/{word2}"

    # Send a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        similarity_score = data['value']
        return similarity_score
    else:
        print(f"Error: Could not retrieve similarity for {word1} and {word2}")
        return None

def calculate_sentence_similarity(sentence1, sentence2):
    # Tokenize the sentences into words
    words1 = word_tokenize(sentence1)
    words2 = word_tokenize(sentence2)

    # Initialize a list to store individual word similarities
    word_similarities = []

    # Calculate similarity for each pair of words
    for word1 in words1:
        for word2 in words2:
            word_similarity = get_word_similarity(word1, word2)
            if word_similarity is not None:
                word_similarities.append(word_similarity)

    # Calculate the overall sentence similarity by averaging word similarities
    if word_similarities:
        sentence_similarity = sum(word_similarities) / len(word_similarities)
    else:
        # Return a low similarity score if no words were found in ConceptNet
        sentence_similarity = 0.0

    return sentence_similarity

# Example usage
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast brown dog jumps over a sleepy fox."
similarity_score = calculate_sentence_similarity(sentence1, sentence2)
print(f"Similarity between sentences:\n'{sentence1}'\nand\n'{sentence2}': {similarity_score}")




# Output:
# Similarity between sentences:
# 'The quick brown fox jumps over the lazy dog.'
# and
# 'A fast brown dog jumps over a sleepy fox.': 0.08977000000000002
