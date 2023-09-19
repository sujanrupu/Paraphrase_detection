from nltk.corpus import wordnet

def get_word_synonyms_from_sent(word, sent):
    word_synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                word_synonyms.append(lemma)
    return word_synonyms

word = "happy"
sent = ['I', 'am', 'glad', 'it', 'was', 'felicitous', '.']
word_synonyms = get_word_synonyms_from_sent(word, sent)
print ("WORD:", word)
print ("SENTENCE:", sent)
print ("SYNONYMS FOR '" + word.upper() + "' FOUND IN THE SENTENCE: " + ", ".join(word_synonyms))