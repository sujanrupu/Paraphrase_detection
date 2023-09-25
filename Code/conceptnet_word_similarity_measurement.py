import requests

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
        print("Error: Could not retrieve similarity.")
        return None

# Example usage
word1 = "cat"
word2 = "dog"
similarity_score = get_word_similarity(word1, word2)
if similarity_score is not None:
    print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")




# Output:
# Similarity between 'cat' and 'dog': 0.558
