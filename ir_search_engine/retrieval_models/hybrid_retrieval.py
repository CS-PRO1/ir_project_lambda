import math
from collections import defaultdict
from typing import List, Tuple

from ir_search_engine.data_processing import TextPreprocessor
from .indexer import InvertedIndex
from .retrieval_model import VectorSpaceModel
from .bert_retrieval import BERTRetrievalModel

class HybridRanker:
    def __init__(self, vsm_inverted_index: InvertedIndex, preprocessor: TextPreprocessor, bert_model: BERTRetrievalModel):
        self.vsm_model = VectorSpaceModel(vsm_inverted_index, preprocessor)
        self.bert_model = bert_model
        if not self.bert_model.documents_text:
            print("Warning: BERT model's documents_text not set. Ensure it's populated for text retrieval.")

    @staticmethod
    def _normalize_scores(scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not scores:
            return []

        doc_ids, raw_scores = zip(*scores)
        
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score == min_score:
            return [(doc_id, 0.5) for doc_id in doc_ids]

        normalized_scores = []
        for i, doc_id in enumerate(doc_ids):
            normalized_score = (raw_scores[i] - min_score) / (max_score - min_score)
            normalized_scores.append((doc_id, normalized_score))
            
        return normalized_scores

    def hybrid_search(self, query_text: str, top_k: int = 10, 
                      vsm_weight: float = 0.3, bert_weight: float = 0.7,
                      top_k_bert_initial: int = 100) -> list[tuple[str, float]]:
        if not (0 <= vsm_weight <= 1 and 0 <= bert_weight <= 1):
            raise ValueError("vsm_weight and bert_weight must be between 0 and 1.")

        print(f"Running Hybrid Search (BERT-first re-ranking)...")

        bert_initial_results = self.bert_model.search(query_text, top_k=top_k_bert_initial)
        
        if not bert_initial_results:
            return []

        documents_for_rerank = []
        
        for doc_id, bert_score in bert_initial_results:
            tfidf_score = self.vsm_model.score_document(query_text, doc_id)
            documents_for_rerank.append((doc_id, bert_score, tfidf_score))

        bert_scores_only = [(doc_id, score) for doc_id, score, _ in documents_for_rerank]
        tfidf_scores_only = [(doc_id, score) for doc_id, _, score in documents_for_rerank]

        normalized_bert_scores = self._normalize_scores(bert_scores_only)
        normalized_tfidf_scores = self._normalize_scores(tfidf_scores_only)

        norm_bert_dict = {doc_id: score for doc_id, score in normalized_bert_scores}
        norm_tfidf_dict = {doc_id: score for doc_id, score in normalized_tfidf_scores}

        fused_scores = []
        for doc_id, _, _ in documents_for_rerank:
            norm_bert = norm_bert_dict.get(doc_id, 0.0)
            norm_tfidf = norm_tfidf_dict.get(doc_id, 0.0)
            
            combined_score = (bert_weight * norm_bert) + (vsm_weight * norm_tfidf)
            fused_scores.append((doc_id, combined_score))

        fused_results = sorted(fused_scores, key=lambda item: item[1], reverse=True)
            
        return fused_results[:top_k]