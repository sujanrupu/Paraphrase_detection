import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define a function to calculate BERT embeddings for a given sentence
def get_bert_embeddings(sentence):
    # Tokenize the sentence
    input_ids = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # Get BERT embeddings
    with torch.no_grad():
        output = model(input_ids)
    # Take the embeddings from the [CLS] token
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

# Calculate the similarity between two sentences
def calculate_similarity(sentence1, sentence2):
    # Get BERT embeddings for the two sentences
    embeddings1 = get_bert_embeddings(sentence1)
    embeddings2 = get_bert_embeddings(sentence2)
    
    # Calculate the cosine similarity between the embeddings
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).item()
    
    return similarity

# Example usage
sentence1 = "The quick brown fox jumps over the lazy dog."
sentence2 = "A fast brown fox leaps over a sleepy canine."
similarity = calculate_similarity(sentence1, sentence2)
print(f"Similarity between Sentence 1 and Sentence 2: {similarity:.4f}")




# Output:

# Similarity between Sentence 1 and Sentence 2: 0.9530
