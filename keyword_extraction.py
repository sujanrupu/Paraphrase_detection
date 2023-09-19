import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('stopwords')
nltk.download('punkt')
def find_keywords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    freq_dist = FreqDist(words)
    keywords = freq_dist.most_common(5)  
    return [keyword[0] for keyword in keywords]
input_text = "India, officially the Republic of India, is a country in South Asia."
result = find_keywords(input_text)
print("Keywords:", result)