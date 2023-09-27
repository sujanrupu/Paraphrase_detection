import torch
from transformers import BertTokenizer, BertModel, pipeline

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define a function to calculate BERT embeddings for a given text
def get_bert_embeddings(text):
    # Tokenize the text
    input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # Get BERT embeddings
    with torch.no_grad():
        output = model(input_ids)
    # Take the embeddings from the [CLS] token
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

# Calculate the syntax and lexical similarity between two words
def calculate_similarity(word1, word2):
    # Get BERT embeddings for the two words
    embeddings1 = get_bert_embeddings(word1)
    embeddings2 = get_bert_embeddings(word2)
    
    # Calculate the cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()
    
    return similarity

# Example usage
word1 = "cat"
word2 = "dog"
similarity = calculate_similarity(word1, word2)
print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")