import math
from collections import defaultdict
import json # Used for saving/loading
import numpy as np # Typically used for vector operations, especially if you have dense vectors

# Assuming these are defined in your project structure
from indexer import InvertedIndex
from preprocessing import TextPreprocessor

class VectorSpaceModel:
    def __init__(self, inverted_index: InvertedIndex, preprocessor: TextPreprocessor):
        self.inverted_index = inverted_index
        self.preprocessor = preprocessor
        # Corrected: Changed .document_lengths to .doc_lengths
        self.total_documents = len(self.inverted_index.doc_lengths)
        self.avg_doc_length = sum(self.inverted_index.doc_lengths.values()) / self.total_documents if self.total_documents > 0 else 0

    def _calculate_tf(self, term_freq):
        """Calculates term frequency (TF) using log normalization."""
        return 1 + math.log10(term_freq) if term_freq > 0 else 0

    def _calculate_idf(self, term):
        """Calculates inverse document frequency (IDF)."""
        doc_freq = len(self.inverted_index.index.get(term, {}))
        return math.log10(self.total_documents / doc_freq) if doc_freq > 0 else 0

    def _calculate_tfidf(self, term_freq, doc_freq):
        """Combines TF and IDF."""
        tf = self._calculate_tf(term_freq)
        idf = math.log10(self.total_documents / doc_freq) if doc_freq > 0 else 0
        return tf * idf

    def _vectorize_query(self, preprocessed_query_terms):
        """Creates a TF-IDF vector for a preprocessed query."""
        query_vector = defaultdict(float)
        query_term_counts = defaultdict(int)
        for term in preprocessed_query_terms:
            query_term_counts[term] += 1
        
        # Calculate TF-IDF for query terms
        for term, count in query_term_counts.items():
            doc_freq = len(self.inverted_index.index.get(term, {}))
            query_vector[term] = self._calculate_tfidf(count, doc_freq)
            
        # Normalize query vector (Euclidean norm)
        magnitude = math.sqrt(sum(val**2 for val in query_vector.values()))
        if magnitude > 0:
            for term in query_vector:
                query_vector[term] /= magnitude
        return query_vector

    def _compute_cosine_similarity(self, query_vector, doc_id):
        """
        Computes cosine similarity between query vector and a document's TF-IDF vector.
        Documents are implicitly represented by their term weights in the inverted index.
        """
        # Corrected: Changed .document_lengths to .doc_lengths
        if doc_id not in self.inverted_index.doc_lengths:
            return 0.0 # Document not found or not indexed

        intersection_terms = set(query_vector.keys()) & set(self.inverted_index.get_document_terms(doc_id))
        
        dot_product = 0.0
        doc_magnitude_sq = 0.0 # Will calculate on the fly for relevant terms
        
        # Calculate dot product and document vector magnitude for terms in intersection
        for term in intersection_terms:
            term_freq_in_doc = self.inverted_index.index[term][doc_id] # Get raw term frequency
            doc_freq = len(self.inverted_index.index.get(term, {})) # Get document frequency for IDF
            
            # Assuming document vector elements are also TF-IDF weights
            doc_tfidf = self._calculate_tfidf(term_freq_in_doc, doc_freq)
            
            dot_product += query_vector[term] * doc_tfidf
            doc_magnitude_sq += doc_tfidf ** 2 # Sum of squares for document vector magnitude

        # Query magnitude is already 1 because it's normalized in _vectorize_query
        doc_magnitude = math.sqrt(doc_magnitude_sq)

        if doc_magnitude == 0: # Avoid division by zero if document vector is all zeros
            return 0.0

        similarity = dot_product / doc_magnitude
        return similarity

    def search(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Performs a TF-IDF search for the given query.

        :param query_text: The user's query string.
        :param top_k: The number of top documents to return.
        :return: A list of (doc_id, score) tuples, sorted by score.
        """
        preprocessed_query = self.preprocessor.preprocess_query(
            query_text, use_stemming=False, use_lemmatization=True, add_ngrams=False
        )
        if not preprocessed_query:
            return []

        query_vector = self._vectorize_query(preprocessed_query)
        
        # Collect candidate documents from the inverted index based on query terms
        candidate_docs = set()
        for term in query_vector.keys():
            if term in self.inverted_index.index:
                candidate_docs.update(self.inverted_index.index[term].keys())

        # Calculate scores for candidate documents
        doc_scores = []
        for doc_id in candidate_docs:
            score = self._compute_cosine_similarity(query_vector, doc_id)
            if score > 0: # Only add documents with a positive similarity score
                doc_scores.append((doc_id, score))

        # Sort documents by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def score_document(self, query_text: str, doc_id: str) -> float:
        """
        Calculates the TF-IDF cosine similarity score for a single document given a query.

        :param query_text: The user's query string.
        :param doc_id: The ID of the document to score.
        :return: The TF-IDF cosine similarity score (float).
        """
        preprocessed_query = self.preprocessor.preprocess_query(
            query_text, use_stemming=False, use_lemmatization=True, add_ngrams=False
        )
        # Corrected: Changed .document_lengths to .doc_lengths
        if not preprocessed_query or doc_id not in self.inverted_index.doc_lengths:
            return 0.0

        query_vector = self._vectorize_query(preprocessed_query)
        
        # We don't need to iterate through all candidates, just score the given doc_id
        score = self._compute_cosine_similarity(query_vector, doc_id)
        return score