import math
from collections import defaultdict
from typing import List, Tuple # For type hints

# Assuming these are available from your project structure
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from retrieval_model import VectorSpaceModel # For TF-IDF
from bert_retrieval import BERTRetrievalModel # For BERT

class HybridRanker:
    def __init__(self, vsm_inverted_index: InvertedIndex, preprocessor: TextPreprocessor, bert_model: BERTRetrievalModel):
        self.vsm_model = VectorSpaceModel(vsm_inverted_index, preprocessor)
        self.bert_model = bert_model
        # Ensure bert_model has access to raw document texts for BERT search results
        if not self.bert_model.documents_text:
            print("Warning: BERT model's documents_text not set. Ensure it's populated for text retrieval.")

    # Keeping _normalize_scores as it will be used for the re-ranking step
    @staticmethod
    def _normalize_scores(scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Performs Min-Max normalization on a list of (doc_id, score) tuples.
        Scores are normalized to the [0, 1] range.
        If all scores are the same, they are all normalized to 0.5 to avoid division by zero.

        :param scores: A list of (doc_id, score) tuples.
        :return: A list of (doc_id, normalized_score) tuples.
        """
        if not scores:
            return []

        doc_ids, raw_scores = zip(*scores)
        
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            # Avoid division by zero; if all scores are the same, normalize to 0.5
            return [(doc_id, 0.5) for doc_id in doc_ids]

        normalized_scores = []
        for i, doc_id in enumerate(doc_ids):
            normalized_score = (raw_scores[i] - min_score) / (max_score - min_score)
            normalized_scores.append((doc_id, normalized_score))
            
        return normalized_scores

    def hybrid_search(self, query_text: str, top_k: int = 10, 
                      vsm_weight: float = 0.3, bert_weight: float = 0.7,
                      top_k_bert_initial: int = 100) -> list[tuple[str, float]]:
        """
        Performs a hybrid search using BERT for initial retrieval and TF-IDF for re-ranking.

        :param query_text: The user's query string.
        :param top_k: The number of top documents to return after re-ranking.
        :param vsm_weight: Weight for VSM (TF-IDF) scores in the re-ranking.
                           Must be between 0 and 1.
        :param bert_weight: Weight for BERT scores in the re-ranking.
                            Must be between 0 and 1.
        :param top_k_bert_initial: The number of top documents to retrieve initially from BERT.
        :return: A list of (doc_id, fused_score) tuples, sorted by fused_score.
        """
        if not (0 <= vsm_weight <= 1 and 0 <= bert_weight <= 1):
            raise ValueError("vsm_weight and bert_weight must be between 0 and 1.")
        
        # Optional: warn if weights don't sum close to 1
        # if not math.isclose(vsm_weight + bert_weight, 1.0):
        #     print("Warning: vsm_weight and bert_weight do not sum to 1.0. This will scale the combined score.")

        print(f"Running Hybrid Search (BERT-first re-ranking)...")

        # Phase 1: BERT Initial Retrieval
        # Get a larger set of initial candidates from BERT, as it's the primary retriever
        bert_initial_results = self.bert_model.search(query_text, top_k=top_k_bert_initial)
        
        if not bert_initial_results:
            return []

        # Prepare data structures for re-ranking
        documents_for_rerank = [] # Stores (doc_id, original_bert_score, tfidf_score)
        
        # Phase 2: TF-IDF Re-scoring for BERT candidates
        # Store BERT scores and calculate TF-IDF scores for these documents
        for doc_id, bert_score in bert_initial_results:
            # Calculate TF-IDF score for this specific document and query
            tfidf_score = self.vsm_model.score_document(query_text, doc_id)
            documents_for_rerank.append((doc_id, bert_score, tfidf_score))

        # Separate scores for normalization
        bert_scores_only = [(doc_id, score) for doc_id, score, _ in documents_for_rerank]
        tfidf_scores_only = [(doc_id, score) for doc_id, _, score in documents_for_rerank]

        # Phase 3: Score Normalization
        # Normalize the scores of the *candidate documents* to a 0-1 range
        normalized_bert_scores = self._normalize_scores(bert_scores_only)
        normalized_tfidf_scores = self._normalize_scores(tfidf_scores_only)

        # Create dictionaries for quick lookup of normalized scores
        norm_bert_dict = {doc_id: score for doc_id, score in normalized_bert_scores}
        norm_tfidf_dict = {doc_id: score for doc_id, score in normalized_tfidf_scores}

        # Phase 4: Weighted Sum Fusion and Re-ranking
        fused_scores = []
        for doc_id, _, _ in documents_for_rerank: # Iterate through the original candidates
            norm_bert = norm_bert_dict.get(doc_id, 0.0)
            norm_tfidf = norm_tfidf_dict.get(doc_id, 0.0)
            
            # Combine normalized scores using weights
            combined_score = (bert_weight * norm_bert) + (vsm_weight * norm_tfidf)
            fused_scores.append((doc_id, combined_score))

        # Phase 5: Final Selection - Sort by combined score in descending order
        fused_results = sorted(fused_scores, key=lambda item: item[1], reverse=True)
            
        return fused_results[:top_k] # Return only the final top_k documents