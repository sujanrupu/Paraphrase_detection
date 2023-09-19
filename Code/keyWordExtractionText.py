import spacy

# Load the spaCy model (you might need to download it using `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

# Function to extract keywords from a sentence
def extract_keywords(sentence):
    keywords = []
    doc = nlp(sentence)
    for token in doc:
        if not token.is_stop and token.is_alpha:
            keywords.append(token.text.lower())
    return keywords

# Read the input file
with open("data.txt", "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Process each sentence and format the output
output = ""
for sentence in sentences:
    sentence = sentence.strip()
    keywords = extract_keywords(sentence)
    keyword_str = ", ".join(keywords)
    output += f"{keyword_str}\n"

# Write the output to a new file
with open("data1.txt", "w", encoding="utf-8") as file:
    file.write(output)

print("Keywords extracted and saved to 'output.txt'.")
