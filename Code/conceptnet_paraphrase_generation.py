import requests
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random

def get_synonyms(word):
    # Get synonyms for a word using NLTK's WordNet
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

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

def rephrase_sentence(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Initialize a list to store rephrased words
    rephrased_words = []

    # Iterate through each word in the sentence
    for word in words:
        # Get synonyms for the word
        synonyms = get_synonyms(word)

        if synonyms:
            # Calculate word similarity scores with each synonym
            similarity_scores = [get_word_similarity(word, synonym) for synonym in synonyms if synonym != word]

            # Select a synonym with the highest similarity score (if any)
            if similarity_scores:
                best_synonym = synonyms[similarity_scores.index(max(similarity_scores))]
                rephrased_words.append(best_synonym)
            else:
                rephrased_words.append(word)
        else:
            rephrased_words.append(word)

    # Join the rephrased words to form a rephrased sentence
    rephrased_sentence = ' '.join(rephrased_words)

    return rephrased_sentence

# Example usage
original_sentence = "The quick brown fox jumps over the lazy dog."
rephrased_sentence = rephrase_sentence(original_sentence)
print(f"Original sentence: {original_sentence}")
print(f"Rephrased sentence: {rephrased_sentence}")




# Output:
# Original sentence: The quick brown fox jumps over the lazy dog.
# Rephrased sentence: The promptly dark-brown slyboots jump over the work-shy bounder .
