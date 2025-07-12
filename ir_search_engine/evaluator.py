import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Any, Optional, Tuple
from tqdm import tqdm

# Ensure these imports are correct based on your project structure
from ir_search_engine.data_loader import Query, Qrel
from ir_search_engine.retrieval_model import VectorSpaceModel
from ir_search_engine.bert_retrieval import BERTRetrievalModel
from ir_search_engine.hybrid_retrieval import HybridRanker
from ir_search_engine.preprocessing import TextPreprocessor
from ir_search_engine.query_optimizer import QueryOptimizer
from ir_search_engine.clusterer import DocumentClusterer

"""
Information Retrieval Evaluation Metrics

This module calculates the following four key metrics for search engine evaluation:

1. Mean Average Precision (MAP): 
   - Average of the precision values after each relevant document is retrieved
   - Measures both precision and recall in a single metric
   - Range: [0, 1], higher is better

2. Recall: 
   - Proportion of relevant documents that were successfully retrieved
   - Measures completeness of retrieval
   - Range: [0, 1], higher is better

3. Precision at 10 (P@10): 
   - Precision of the top 10 retrieved documents
   - Measures precision at a fixed cutoff
   - Range: [0, 1], higher is better

4. Mean Reciprocal Rank (MRR): 
   - Average of the reciprocal of the rank of the first relevant document
   - Measures how early the first relevant document appears
   - Range: [0, 1], higher is better
"""

# --- Metric Calculation Functions (KEEP THESE AS THEY ARE) ---
def calculate_precision(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    # ... (existing code) ...
    # (Assuming the rest of the metric functions like calculate_recall, calculate_f1,
    #  calculate_ndcg, calculate_map are also correctly defined here)
    relevant_retrieved = 0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
            relevant_retrieved += 1
    return relevant_retrieved / k if k > 0 else 0.0

def calculate_recall(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    num_relevant_docs = sum(1 for rel in relevant_docs.values() if rel > 0)
    if num_relevant_docs == 0:
        return 1.0 # If there are no relevant documents, perfect recall is achieved vacuously
    
    relevant_retrieved = 0
    for doc_id in retrieved_doc_ids[:k]:
        if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
            relevant_retrieved += 1
    return relevant_retrieved / num_relevant_docs

def calculate_f1(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    p = calculate_precision(retrieved_doc_ids, relevant_docs, k)
    r = calculate_recall(retrieved_doc_ids, relevant_docs, k)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)

def calculate_dcg(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        relevance = relevant_docs.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2) # i+1 is rank, so i+2 for log2(rank+1)
    return dcg

def calculate_idcg(relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    # Sort relevant documents by relevance score in descending order
    ideal_relevances = sorted([rel for rel in relevant_docs.values() if rel > 0], reverse=True)
    ideal_dcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        ideal_dcg += relevance / np.log2(i + 2)
    return ideal_dcg

def calculate_ndcg(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    dcg = calculate_dcg(retrieved_doc_ids, relevant_docs, k)
    idcg = calculate_idcg(relevant_docs, k)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_ap(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int]) -> float:
    # Calculate Average Precision
    precisions = []
    num_relevant_found = 0
    
    # Filter relevant_docs to only include truly relevant ones (relevance > 0)
    true_relevant_doc_ids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    
    if not true_relevant_doc_ids:
        return 0.0 # No relevant documents for this query, AP is 0
    
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in true_relevant_doc_ids:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / (i + 1)
            precisions.append(precision_at_k)
    
    if not precisions:
        return 0.0 # No relevant documents retrieved
    
    return sum(precisions) / len(true_relevant_doc_ids)

def calculate_map(retrieved_docs_for_query: List[Tuple[Union[int, str], float]], relevant_docs: Dict[Union[int, str], int]) -> float:
    return calculate_ap(retrieved_docs_for_query, relevant_docs)

def calculate_reciprocal_rank(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int]) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single query.
    RR = 1/rank_of_first_relevant_document
    If no relevant document is found, RR = 0
    """
    # Filter relevant_docs to only include truly relevant ones (relevance > 0)
    true_relevant_doc_ids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    
    if not true_relevant_doc_ids:
        return 0.0 # No relevant documents for this query
    
    # Find the rank of the first relevant document (1-indexed)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in true_relevant_doc_ids:
            return 1.0 / (i + 1)  # i+1 because rank is 1-indexed
    
    return 0.0 # No relevant document found in retrieved results


# --- Main Evaluation Function ---
def evaluate_models(
    models: Dict[str, Union[VectorSpaceModel, BERTRetrievalModel, HybridRanker]],
    queries: Dict[Union[int, str], Query],
    qrels: Dict[Union[int, str], Dict[Union[int, str], int]],
    raw_documents_dict: Dict[Union[int, str], str], # Needed for PRF document content
    preprocessor: TextPreprocessor, # Needed for PRF text processing
    query_optimizer: QueryOptimizer, # Needed for PRF logic
    k_values: List[int] = [10],  # Default to [10] for internal calculations
    document_clusterer: Optional[DocumentClusterer] = None, # Needed for clustered BERT eval
    use_clustering_for_bert: bool = False,
    use_prf: bool = False,
    prf_initial_model_name: Optional[str] = None, # e.g., 'TF-IDF', 'BERT'
    prf_top_n_docs: int = 5,
    prf_num_expansion_terms: int = 3,
    prf_final_model_name: Optional[str] = None # New: Which model to use for the final search after PRF
) -> Dict[str, Dict[str, Any]]:
    
    all_results = defaultdict(lambda: defaultdict(list)) # Stores (doc_id, score) for each query-model combination
    
    # 1. Evaluate existing models
    for model_name, model_instance in models.items():
        if model_instance is None:
            print(f"Skipping evaluation for {model_name}: model not initialized.")
            continue
        
        print(f"Evaluating {model_name}...")
        for query_id, query_obj in tqdm(queries.items(), desc=f"Evaluating {model_name}"):
            # All search methods (VSM, BERT, Hybrid) should accept a string query and return List[Tuple[doc_id, score]]
            if model_name.lower() == 'hybrid':
                retrieved_docs = model_instance.hybrid_search(
                    query_obj.text, 
                    top_k=max(k_values) * 2, # Fetch more just in case to cover all k_values
                    vsm_weight=0.1, 
                    bert_weight=0.9, 
                    top_k_bert_initial=max(k_values) * 5 # Enough initial BERT docs for hybrid
                )
            else:
                retrieved_docs = model_instance.search(query_obj.text, top_k=max(k_values) * 2) # Fetch more just in case

            all_results[model_name][query_id] = retrieved_docs

    # 2. Evaluate with Clustering (if enabled)
    if use_clustering_for_bert and document_clusterer and 'BERT' in models:
        bert_model_instance = models['BERT']
        if not bert_model_instance:
            print("Cannot perform clustered BERT evaluation: BERT model not available.")
        else:
            print(f"Evaluating BERT with Clustering...")
            model_name_clustered = "BERT + Clustering"
            for query_id, query_obj in tqdm(queries.items(), desc=f"Evaluating {model_name_clustered}"):
                query_embedding = bert_model_instance.encode_query(query_obj.text)
                nearest_cluster_id = document_clusterer.find_nearest_cluster(query_embedding)
                cluster_doc_ids = document_clusterer.get_documents_in_cluster(nearest_cluster_id)
                
                # Use the modified BERT search that accepts candidate_doc_ids
                retrieved_docs_clustered = bert_model_instance.search(
                    query_obj.text, 
                    top_k=max(k_values) * 2, # Fetch enough for all k_values
                    candidate_doc_ids=cluster_doc_ids
                )
                all_results[model_name_clustered][query_id] = retrieved_docs_clustered

    # 3. Evaluate with PRF (if enabled)
    if use_prf and query_optimizer and prf_initial_model_name:
        initial_retrieval_model = models.get(prf_initial_model_name)
        if not initial_retrieval_model:
            print(f"Cannot perform PRF evaluation: Initial model '{prf_initial_model_name}' not available or initialized.")
        else:
            final_retrieval_model_instance = models.get(prf_final_model_name if prf_final_model_name else prf_initial_model_name)
            if not final_retrieval_model_instance:
                print(f"Cannot perform PRF evaluation: Final model '{prf_final_model_name or prf_initial_model_name}' not available or initialized.")
            else:
                print(f"Evaluating {prf_initial_model_name} with PRF (final search with {prf_final_model_name or prf_initial_model_name})...")
                model_name_prf = f"{prf_initial_model_name} + PRF (final: {prf_final_model_name or prf_initial_model_name})"
                
                for query_id, query_obj in tqdm(queries.items(), desc=f"Evaluating {model_name_prf}"):
                    expanded_query = query_optimizer.expand_query_with_prf(
                        original_query=query_obj.text,
                        retrieval_model=initial_retrieval_model,
                        raw_documents_dict=raw_documents_dict,
                        top_n_docs_for_prf=prf_top_n_docs,
                        num_expansion_terms=prf_num_expansion_terms
                    )
                    
                    # Perform search with the expanded query using the specified final model
                    if prf_final_model_name and prf_final_model_name.lower() == 'hybrid':
                        retrieved_docs_prf = final_retrieval_model_instance.hybrid_search(
                            expanded_query, 
                            top_k=max(k_values) * 2,
                            vsm_weight=0.1, 
                            bert_weight=0.9, 
                            top_k_bert_initial=max(k_values) * 5
                        )
                    else:
                        retrieved_docs_prf = final_retrieval_model_instance.search(expanded_query, top_k=max(k_values) * 2)
                    
                    all_results[model_name_prf][query_id] = retrieved_docs_prf


    # 4. Calculate metrics for all collected results
    final_metrics = {}
    for model_name, query_results in all_results.items():
        if not query_results: 
            continue
        
        # Focus on the specific metrics requested: MAP, Recall, Precision at 10, MRR
        map_scores = []
        recall_scores = []  # Overall recall (not at specific k)
        precision_at_10_scores = []
        rr_scores = []  # Reciprocal Rank scores for MRR

        # Iterate through all queries *defined in the qrels* to ensure metrics are calculated for all
        # queries that have relevance judgments, even if they return no documents.
        for query_id in qrels.keys(): # Use qrels.keys() as the source of truth for queries to evaluate
            relevant_docs_for_query = qrels.get(query_id, {})
            retrieved_docs_for_query = query_results.get(query_id, [])

            # Handle cases where there are no relevant documents or no retrieved documents
            num_true_relevant = sum(1 for rel in relevant_docs_for_query.values() if rel > 0)
            
            if num_true_relevant == 0:
                # If no relevant documents exist for this query
                if not retrieved_docs_for_query:
                    # Skip for MAP/MRR calculation (no relevant docs and no retrieved)
                    pass 
                else: # Retrieved docs but no relevant ones
                    precision_at_10_scores.append(0.0)
                    recall_scores.append(1.0) # If no relevant docs, recall is 1.0
                    rr_scores.append(0.0) # No relevant document found
                map_scores.append(0.0)
                continue # Move to next query

            if not retrieved_docs_for_query: # Relevant docs exist, but nothing was retrieved
                precision_at_10_scores.append(0.0)
                recall_scores.append(0.0)
                rr_scores.append(0.0) # No relevant document found
                map_scores.append(0.0)
                continue

            retrieved_doc_ids_only = [doc_id for doc_id, _ in retrieved_docs_for_query]

            # Calculate the specific metrics requested
            # 1. MAP (Mean Average Precision)
            map_scores.append(calculate_map(retrieved_doc_ids_only, relevant_docs_for_query))
            
            # 2. Recall (overall recall, not at specific k)
            recall_scores.append(calculate_recall(retrieved_doc_ids_only, relevant_docs_for_query, len(retrieved_doc_ids_only)))
            
            # 3. Precision at 10
            precision_at_10_scores.append(calculate_precision(retrieved_doc_ids_only, relevant_docs_for_query, 10))
            
            # 4. Reciprocal Rank (for MRR)
            rr_scores.append(calculate_reciprocal_rank(retrieved_doc_ids_only, relevant_docs_for_query))

        # Calculate averages for the specific metrics
        avg_map = np.mean(map_scores) if map_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        avg_precision_at_10 = np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0
        avg_rr = np.mean(rr_scores) if rr_scores else 0.0  # This is MRR (Mean Reciprocal Rank)

        final_metrics[model_name] = {
            'map': avg_map,  # Mean Average Precision
            'recall': avg_recall,  # Overall Recall
            'precision_at_10': avg_precision_at_10,  # Precision at 10
            'mrr': avg_rr  # Mean Reciprocal Rank
        }
    
    return final_metrics