import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text

# Function to calculate semantic similarity using ConceptNet
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

# Function to calculate accuracy for a range of lines
def calculate_accuracy_range(file1_lines, file2_lines, start, end):
    similarity_scores = []

    for i in range(start, end):
        line1 = file1_lines[i]
        line2 = file2_lines[i]

        similarity = calculate_semantic_similarity(line1, line2)
        similarity_scores.append(similarity)

    return similarity_scores

# Read lines from both text files
with open('data1.txt', 'r', encoding='utf-8') as file1, open('data2.txt', 'r', encoding='utf-8') as file2:
    file1_lines = file1.readlines()
    file2_lines = file2.readlines()

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
    futures = [executor.submit(calculate_accuracy_range, file1_lines, file2_lines, start, end) for start, end in chunks]

    # Gather results from completed tasks
    similarity_scores = []
    for future in as_completed(futures):
        similarity_scores.extend(future.result())

# Calculate the average similarity score
average_similarity = sum(similarity_scores) / len(similarity_scores)

print(f'Semantic Similarity Accuracy: {average_similarity}')


## Semantic Similarity Accuracy: 68.6315%
