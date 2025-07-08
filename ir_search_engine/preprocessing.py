import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union
from tqdm import tqdm # Import tqdm

# Ensure NLTK data is downloaded (run these lines once if you encounter errors)
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

    def preprocess_document(self, text: str) -> List[str]:
        """
        Applies basic preprocessing steps to a single document text.
        Steps: lowercase, remove punctuation, tokenize, remove non-alphanumeric, remove stopwords.
        """
        # Lowercasing
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenization
        tokens = word_tokenize(text)

        # Remove non-alphanumeric tokens and stopwords
        processed_tokens = []
        for token in tokens:
            # --- FIX IS HERE: Changed .isalpha() to .isalnum() ---
            if token.isalnum() and token not in self.stop_words: 
                processed_tokens.append(token)
        
        return processed_tokens

    def preprocess_query(self, text: str, use_stemming: bool = False, use_lemmatization: bool = True, add_ngrams: bool = False) -> List[str]:
        """
        Preprocesses a single query string.
        Applies selected transformations (lemmatization/stemming, n-grams) based on flags.
        """
        tokens = self.preprocess_document(text) # Basic preprocessing first

        transformed_tokens = []
        for token in tokens:
            if use_lemmatization:
                transformed_tokens.append(self.lemmatizer.lemmatize(token))
            elif use_stemming:
                transformed_tokens.append(self.stemmer.stem(token))
            else:
                transformed_tokens.append(token)

        # Add n-grams if requested (only bigrams and trigrams for simplicity)
        if add_ngrams:
            # Bigrams
            for i in range(len(transformed_tokens) - 1):
                transformed_tokens.append(f"{transformed_tokens[i]}_{transformed_tokens[i+1]}")
            # Trigrams
            for i in range(len(transformed_tokens) - 2):
                transformed_tokens.append(f"{transformed_tokens[i]}_{transformed_tokens[i+1]}_{transformed_tokens[i+2]}")

        return transformed_tokens

    def preprocess_documents(self, documents_texts: List[str], use_stemming: bool = False, use_lemmatization: bool = True, add_ngrams: bool = False, desc: str = "Preprocessing documents") -> List[List[str]]:
        """
        Preprocesses a list of document texts.

        :param documents_texts: A list of raw document strings.
        :param use_stemming: Whether to apply stemming.
        :param use_lemmatization: Whether to apply lemmatization.
        :param add_ngrams: Whether to add n-grams (bigrams and trigrams).
        :param desc: Description for the tqdm progress bar.
        :return: A list of lists, where each inner list contains the preprocessed terms for a document.
        """
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

# Example Usage (for direct testing of preprocessing.py)
if __name__ == "__main__":
    print("--- Testing TextPreprocessor (Standalone) ---")

    preprocessor = TextPreprocessor(language='english')

    # Test single document preprocessing
    doc_text = "This is an example sentence, with some CAPITAL words and numbers like 123! Dogs are running."
    processed_doc = preprocessor.preprocess_document(doc_text)
    print(f"\nOriginal Document: '{doc_text}'")
    print(f"Processed Document (basic): {processed_doc}")

    # Test query preprocessing
    query_text = "What are the best running shoes? 1945."
    stemmed_query = preprocessor.preprocess_query(query_text, use_stemming=True, use_lemmatization=False)
    lemmatized_query = preprocessor.preprocess_query(query_text, use_stemming=False, use_lemmatization=True)
    lemmatized_ngrams_query = preprocessor.preprocess_query(query_text, use_lemmatization=True, add_ngrams=True)

    print(f"\nOriginal Query: '{query_text}'")
    print(f"Stemmed Query: {stemmed_query}")
    print(f"Lemmatized Query: {lemmatized_query}")
    print(f"Lemmatized Query with N-grams: {lemmatized_ngrams_query}")

    # Test batch document preprocessing
    documents_for_batch = [
        "The quick brown fox jumps over the lazy dog.",
        "Dogs are loyal animals, and cats are independent.",
        "A fast car race is exciting."
    ]
    print("\nBatch Preprocessing Documents:")
    processed_batch = preprocessor.preprocess_documents(documents_for_batch, use_lemmatization=True, desc="Batch Preprocessing Test")
    for i, doc in enumerate(processed_batch):
        print(f"Doc {i+1}: {doc}")
    
    # Test a document with special characters and numbers
    special_doc = "Hello world! This is a test with 123 numbers and @symbols. Also, a date 1999-01-01"
    processed_special_doc = preprocessor.preprocess_document(special_doc)
    print(f"\nSpecial Chars Document: '{special_doc}'")
    print(f"Processed Special Chars Document: {processed_special_doc}")