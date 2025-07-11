import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union
from tqdm import tqdm # Import tqdm

# For spelling correction
from spellchecker import SpellChecker # <--- ADD THIS IMPORT

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
        
        # Initialize SpellChecker
        self.spell = SpellChecker(language='en') # <--- ADD THIS LINE
        # You can optionally build your vocabulary from your dataset
        # For antique_train, you could load all terms from your VSM's inverted index
        # Example (you'd need to pass the InvertedIndex or its vocabulary):
        # self.spell.word_frequency.load_words(your_inverted_index.get_all_terms())

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

    def correct_query_spelling(self, query: str) -> str: # <--- ADD THIS NEW METHOD
        """
        Corrects the spelling of individual words in a query string.
        Assumes basic tokenization and lowercasing.
        """
        # It's usually best to correct spelling *before* heavy preprocessing like stemming/lemmatization
        # because the spell checker works best on "real" words.
        # We can reuse preprocess_document for simple tokenization/lowercasing for the spell checker.
        tokens = self.preprocess_document(query) # Get basic tokens (lowercase, no punctuation, no stopwords)
        corrected_tokens = []
        for token in tokens:
            corrected_token = self.spell.correction(token)
            if corrected_token: # Ensure a correction was found
                corrected_tokens.append(corrected_token)
            else: # If no correction, keep original token (or decide to drop it if it's very bad)
                corrected_tokens.append(token)
        
        return " ".join(corrected_tokens) # Join back into a string