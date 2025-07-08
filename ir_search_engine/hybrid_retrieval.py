import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Import the core components
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from retrieval_model import VectorSpaceModel # Your TF-IDF based model
from bert_retrieval import BERTRetrievalModel # Your BERT embedding based model

class HybridRanker:
    def __init__(self, index: InvertedIndex, preprocessor: TextPreprocessor, bert_model: BERTRetrievalModel):
        """
        Initializes the HybridRanker with instances of the individual models.

        :param index: An instance of the InvertedIndex (needed by VSM).
        :param preprocessor: An instance of the TextPreprocessor.
        :param bert_model: An instance of the BERTRetrievalModel.
        """
        self.vsm = VectorSpaceModel(index, preprocessor)
        self.bert_model = bert_model
        self.preprocessor = preprocessor # Keep a reference for consistency

        # Default weights for weighted sum fusion (can be tuned)
        self.tfidf_weight = 0.5
        self.bert_weight = 0.5

    def _normalize_scores(self, results: list) -> dict:
        """
        Normalizes scores from a list of (doc_id, score) tuples to a 0-1 range.
        Uses min-max scaling.

        :param results: A list of (doc_id, score) tuples.
        :return: A dictionary {doc_id: normalized_score}.
        """
        if not results:
            return {}
        
        scores = [score for doc_id, score in results]
        min_score = min(scores)
        max_score = max(scores)

        normalized_scores = {}
        if max_score == min_score: # Avoid division by zero if all scores are identical
            # If all scores are the same, they should be treated equally high (e.g., 1.0)
            # or handle as a special case if they are all 0.
            if max_score == 0: # If all scores are zero, normalize to 0
                for doc_id, _ in results:
                    normalized_scores[doc_id] = 0.0
            else: # All non-zero scores are identical, normalize to 1.0
                for doc_id, _ in results:
                    normalized_scores[doc_id] = 1.0
        else:
            for doc_id, score in results:
                normalized_scores[doc_id] = (score - min_score) / (max_score - min_score)
        
        return normalized_scores

    def _reciprocal_rank_fusion(self, rank_lists: list, k=60) -> list:
        """
        Applies Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
        Robust to different scoring scales and doesn't require score normalization.

        :param rank_lists: A list of ranked lists. Each inner list is [(doc_id, score), ...].
                           The score in the inner list is used for original ranking, but RRF uses rank.
        :param k: A constant (typically 60) to downweight lower ranks.
        :return: A list of (doc_id, fused_score) tuples, sorted by fused_score descending.
        """
        fused_scores = defaultdict(float)
        
        # Iterate through each ranked list from different models
        for rank_list in rank_lists:
            for rank, (doc_id, _) in enumerate(rank_list): # _ is the original score, not used in RRF score calculation
                fused_scores[doc_id] += 1 / (k + rank + 1) # rank is 0-indexed, so +1 for 1-based rank
        
        # Convert to list and sort
        sorted_fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused_scores

    def hybrid_search(self, query_text: str, top_k: int = 10, 
                      fusion_method: str = 'rrf', 
                      vsm_top_k: int = 50, # How many results each individual model should return
                      bert_top_k: int = 50, # for the fusion stage
                      **kwargs) -> list:
        """
        Executes a hybrid search using both VSM (TF-IDF) and BERT models.

        :param query_text: The raw query string.
        :param top_k: The final number of top documents to return after fusion.
        :param fusion_method: 'rrf' for Reciprocal Rank Fusion, 'weighted_sum' for weighted sum.
        :param vsm_top_k: Number of VSM results to consider for fusion.
        :param bert_top_k: Number of BERT results to consider for fusion.
        :param kwargs: Additional arguments passed to VSM search (e.g., add_ngrams for query preprocessing).
        :return: A list of (doc_id, fused_score) tuples, sorted by score descending.
        """
        # Determine if BERT model has indexed documents
        run_bert = (self.bert_model.document_embeddings_matrix is not None and len(self.bert_model.doc_id_map) > 0)
        
        # Run both searches in parallel
        vsm_results = []
        bert_results = []

        with ThreadPoolExecutor(max_workers=2 if run_bert else 1) as executor:
            # Submit VSM search. VSM's search might print "No docs found" internally.
            # We'll rely on its return value (empty list) to detect if it found anything.
            vsm_future = executor.submit(self.vsm.search, query_text, vsm_top_k, **kwargs)
            
            if run_bert:
                bert_future = executor.submit(self.bert_model.search, query_text, bert_top_k)
            
            vsm_results = vsm_future.result()
            if run_bert:
                bert_results = bert_future.result()

        # Handle cases where no results are found by either model
        if not vsm_results and not bert_results:
            print(f"No documents found by either VSM or BERT for query '{query_text}'. Returning empty results.")
            return []

        fused_ranking = []
        if fusion_method == 'rrf':
            all_rank_lists = []
            if vsm_results:
                all_rank_lists.append(vsm_results)
            if bert_results:
                all_rank_lists.append(bert_results)
            
            if all_rank_lists: # Only fuse if there's at least one non-empty list
                fused_ranking = self._reciprocal_rank_fusion(all_rank_lists)
        
        elif fusion_method == 'weighted_sum':
            # This method inherently requires both models to contribute meaningfully for best results.
            # If only one model has results, its results will be scaled by its weight.
            # If bert_results is empty, the normalization of bert_scores will be empty, and it won't contribute.
            
            normalized_vsm_scores = self._normalize_scores(vsm_results)
            normalized_bert_scores = self._normalize_scores(bert_results)

            combined_scores = defaultdict(float)

            # Combine scores from VSM
            for doc_id, score in normalized_vsm_scores.items():
                combined_scores[doc_id] += score * self.tfidf_weight
            
            # Combine scores from BERT
            for doc_id, score in normalized_bert_scores.items():
                combined_scores[doc_id] += score * self.bert_weight

            fused_ranking = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # A final check for weighted sum: if after combining, all scores are zero or too low
            if not fused_ranking or (fused_ranking and fused_ranking[0][1] < 1e-6): # threshold for "meaningful" score
                 print(f"Hybrid search (weighted_sum) resulted in no meaningful scores for query '{query_text}'. Returning empty results.")
                 return []

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}. Choose 'rrf' or 'weighted_sum'.")

        return fused_ranking[:top_k]

# Example Usage
if __name__ == "__main__":
    print("--- Testing HybridRanker ---")

    # Sample documents (same as used previously)
    sample_raw_documents = {
        0: "The quick brown fox jumps over the lazy dog.",
        1: "Never jump over a lazy dog, it might bark.",
        2: "Foxes are cunning, dogs are loyal. The dog is brown."
    }

    # 1. Initialize Preprocessor
    preprocessor = TextPreprocessor(language='english')
    print("\nInitializing TextPreprocessor.")

    # 2. Build Inverted Index (for VSM)
    print("\nBuilding Inverted Index for VSM...")
    ordered_doc_ids_for_tfidf = sorted(sample_raw_documents.keys())
    ordered_doc_texts_for_tfidf = [sample_raw_documents[doc_id] for doc_id in ordered_doc_ids_for_tfidf]
    preprocessed_output_list = preprocessor.preprocess_documents(
        ordered_doc_texts_for_tfidf, use_stemming=False, use_lemmatization=True, add_ngrams=False
    )
    index = InvertedIndex()
    index.build_index(preprocessed_output_list) 
    print(f"Inverted Index built with {index.get_total_documents()} documents.")

    # 3. Initialize and Index documents for BERT Model
    print("\nInitializing and Indexing documents for BERT Model...")
    bert_model = BERTRetrievalModel()
    bert_model.index_documents(sample_raw_documents)
    print(f"BERT Model indexed {len(bert_model.doc_id_map)} documents.")

    # 4. Initialize the HybridRanker
    hybrid_ranker = HybridRanker(index, preprocessor, bert_model)
    print("\nHybridRanker initialized.")

    # --- Test Hybrid Search with RRF ---
    query_text_1 = "lazy puppy" # Mix of exact and semantic
    print(f"\n--- Hybrid Search (RRF) for: '{query_text_1}' ---")
    results_rrf = hybrid_ranker.hybrid_search(query_text_1, top_k=3, fusion_method='rrf')
    if results_rrf:
        print("Top results (RRF):")
        for doc_id, score in results_rrf:
            # We use bert_model.documents_text to get original text, as it stores it
            print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Original Text: '{bert_model.documents_text.get(doc_id, 'N/A')}'")
    else:
        print("No results found.")

    # --- Test Hybrid Search with Weighted Sum ---
    query_text_2 = "cunning dog" # Mix of exact and semantic
    print(f"\n--- Hybrid Search (Weighted Sum) for: '{query_text_2}' ---")
    results_weighted = hybrid_ranker.hybrid_search(query_text_2, top_k=3, fusion_method='weighted_sum')
    if results_weighted:
        print("Top results (Weighted Sum):")
        for doc_id, score in results_weighted:
            print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Original Text: '{bert_model.documents_text.get(doc_id, 'N/A')}'")
    else:
        print("No results found.")

    # --- Test query with no matches ---
    query_text_3 = "extraterrestrial life"
    print(f"\n--- Hybrid Search (RRF) for: '{query_text_3}' ---")
    results_no_match = hybrid_ranker.hybrid_search(query_text_3, top_k=3, fusion_method='rrf')
    if results_no_match:
        print("Top results (RRF):")
        for doc_id, score in results_no_match:
            print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Original Text: '{bert_model.documents_text.get(doc_id, 'N/A')}'")
    else:
        print("No results found for this query.")

    print("\n--- End of HybridRanker Testing ---")