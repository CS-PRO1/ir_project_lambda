import math
from collections import defaultdict
import json
import numpy as np

from .indexer import InvertedIndex
from ir_search_engine.data_processing import TextPreprocessor

class VectorSpaceModel:
    def __init__(self, inverted_index: InvertedIndex, preprocessor: TextPreprocessor):
        self.inverted_index = inverted_index
        self.preprocessor = preprocessor
        self.total_documents = len(self.inverted_index.doc_lengths)
        self.avg_doc_length = sum(self.inverted_index.doc_lengths.values()) / self.total_documents if self.total_documents > 0 else 0

    def _calculate_tf(self, term_freq):
        return 1 + math.log10(term_freq) if term_freq > 0 else 0

    def _calculate_idf(self, term):
        doc_freq = len(self.inverted_index.index.get(term, {}))
        return math.log10(self.total_documents / doc_freq) if doc_freq > 0 else 0

    def _calculate_tfidf(self, term_freq, doc_freq):
        tf = self._calculate_tf(term_freq)
        idf = math.log10(self.total_documents / doc_freq) if doc_freq > 0 else 0
        return tf * idf

    def _vectorize_query(self, preprocessed_query_terms):
        query_vector = defaultdict(float)
        query_term_counts = defaultdict(int)
        for term in preprocessed_query_terms:
            query_term_counts[term] += 1
        
        for term, count in query_term_counts.items():
            doc_freq = len(self.inverted_index.index.get(term, {}))
            query_vector[term] = self._calculate_tfidf(count, doc_freq)
            
        magnitude = math.sqrt(sum(val**2 for val in query_vector.values()))
        if magnitude > 0:
            for term in query_vector:
                query_vector[term] /= magnitude
        return query_vector

    def _compute_cosine_similarity(self, query_vector, doc_id):
        if doc_id not in self.inverted_index.doc_lengths:
            return 0.0

        intersection_terms = set(query_vector.keys()) & set(self.inverted_index.get_document_terms(doc_id))
        
        dot_product = 0.0
        doc_magnitude_sq = 0.0
        
        for term in intersection_terms:
            term_freq_in_doc = self.inverted_index.index[term][doc_id]
            doc_freq = len(self.inverted_index.index.get(term, {}))
            
            doc_tfidf = self._calculate_tfidf(term_freq_in_doc, doc_freq)
            
            dot_product += query_vector[term] * doc_tfidf
            doc_magnitude_sq += doc_tfidf ** 2

        doc_magnitude = math.sqrt(doc_magnitude_sq)

        if doc_magnitude == 0:
            return 0.0

        similarity = dot_product / doc_magnitude
        return similarity

    def search(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        preprocessed_query = self.preprocessor.preprocess_query(
            query_text, use_stemming=False, use_lemmatization=True, add_ngrams=False
        )
        if not preprocessed_query:
            return []

        query_vector = self._vectorize_query(preprocessed_query)
        
        candidate_docs = set()
        for term in query_vector.keys():
            if term in self.inverted_index.index:
                candidate_docs.update(self.inverted_index.index[term].keys())

        doc_scores = []
        for doc_id in candidate_docs:
            score = self._compute_cosine_similarity(query_vector, doc_id)
            if score > 0:
                doc_scores.append((doc_id, score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def score_document(self, query_text: str, doc_id: str) -> float:
        preprocessed_query = self.preprocessor.preprocess_query(
            query_text, use_stemming=False, use_lemmatization=True, add_ngrams=False
        )
        if not preprocessed_query or doc_id not in self.inverted_index.doc_lengths:
            return 0.0

        query_vector = self._vectorize_query(preprocessed_query)
        score = self._compute_cosine_similarity(query_vector, doc_id)
        return score