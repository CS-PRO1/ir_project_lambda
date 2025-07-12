import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union
from tqdm import tqdm
from spellchecker import SpellChecker

try:
    import nltk
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    import nltk
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker(language='en')

    def preprocess_document(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                processed_tokens.append(token)
        return processed_tokens

    def preprocess_query(self, text: str, use_stemming: bool = False, use_lemmatization: bool = True, add_ngrams: bool = False) -> List[str]:
        tokens = self.preprocess_document(text)

        transformed_tokens = []
        for token in tokens:
            if use_lemmatization:
                transformed_tokens.append(self.lemmatizer.lemmatize(token))
            elif use_stemming:
                transformed_tokens.append(self.stemmer.stem(token))
            else:
                transformed_tokens.append(token)

        if add_ngrams:
            for i in range(len(transformed_tokens) - 1):
                transformed_tokens.append(f"{transformed_tokens[i]}_{transformed_tokens[i+1]}")
            for i in range(len(transformed_tokens) - 2):
                transformed_tokens.append(f"{transformed_tokens[i]}_{transformed_tokens[i+1]}_{transformed_tokens[i+2]}")

        return transformed_tokens

    def preprocess_documents(self, documents_texts: List[str], use_stemming: bool = False, use_lemmatization: bool = True, add_ngrams: bool = False, desc: str = "Preprocessing documents") -> List[List[str]]:
        preprocessed_results = []
        for text in tqdm(documents_texts, desc=desc):
            processed_tokens_for_doc = self.preprocess_query(
                text,
                use_stemming=use_stemming,
                use_lemmatization=use_lemmatization,
                add_ngrams=add_ngrams
            )
            preprocessed_results.append(processed_tokens_for_doc)
        return preprocessed_results

    def correct_query_spelling(self, query: str) -> str:
        tokens = self.preprocess_document(query)
        corrected_tokens = []
        for token in tokens:
            corrected_token = self.spell.correction(token)
            if corrected_token:
                corrected_tokens.append(corrected_token)
            else:
                corrected_tokens.append(token)
        return " ".join(corrected_tokens)