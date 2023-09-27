import torch
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import wordnet
import random

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForMaskedLM.from_pretrained(model_name)

# Function to get synonyms from WordNet
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Function to generate paraphrases for a sentence
def generate_paraphrases(sentence, num_paraphrases=5):
    tokenized_sentence = tokenizer.tokenize(sentence)
    paraphrases = []

    for _ in range(num_paraphrases):
        paraphrase_words = []

        for word in tokenized_sentence:
            # Get synonyms for the word from WordNet
            synonyms = get_synonyms(word)

            if synonyms:
                # Replace the word with a random synonym
                paraphrase_word = random.choice(synonyms)
                paraphrase_words.append(paraphrase_word)
            else:
                paraphrase_words.append(word)

        # Convert the paraphrase words back to text
        paraphrase_sentence = ' '.join(paraphrase_words)
        paraphrases.append(paraphrase_sentence)

    return paraphrases

# Example usage
input_sentence = "The quick brown fox jumps over the lazy dog."
paraphrases = generate_paraphrases(input_sentence, num_paraphrases=5)

print("Original Sentence:")
print(input_sentence)
print("\nGenerated Paraphrases:")
for i, paraphrase in enumerate(paraphrases):
    print(f"{i+1}. {paraphrase}")
