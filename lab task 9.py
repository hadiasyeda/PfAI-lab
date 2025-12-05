import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
text = "Natural Language Processing is amazing. It allows machines to understand human language!"
sentences = sent_tokenize(text)
print("Sentence Tokenization:")
print(sentences)

words = word_tokenize(text)
print("\nWord Tokenization:")
print(words)
