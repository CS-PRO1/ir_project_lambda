import time
from collections import defaultdict
import math # Import math for log2 in NDCG calculation
# CORRECTED IMPORT: Added Tuple
from typing import Dict, List, Set, Tuple, Union
from tqdm import tqdm
# Assuming these are available from your project structure
from retrieval_model import VectorSpaceModel
from bert_retrieval import BERTRetrievalModel
from hybrid_retrieval import HybridRanker
from data_loader import Query # Assuming Query class is in data_loader or defined elsewhere

class Evaluator:
    def __init__(self):
        pass # No specific initialization needed for the evaluator itself

    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List[Tuple[str, float]], qrels: Dict[str, int], k: int) -> float:
        """
        Calculates Precision@K.

        :param retrieved_docs: A list of (doc_id, score) tuples, ranked.
        :param qrels: A dictionary of relevant document IDs and their relevance scores for a query.
        :param k: The number of top documents to consider.
        :return: Precision@K value.
        """
        if k <= 0 or not retrieved_docs:
            return 0.0

        relevant_retrieved = 0
        for i in range(min(k, len(retrieved_docs))):
            doc_id = retrieved_docs[i][0]
            if doc_id in qrels and qrels[doc_id] > 0: # Assuming relevance score > 0 means relevant
                relevant_retrieved += 1
        return relevant_retrieved / k

    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List[Tuple[str, float]], qrels: Dict[str, int], k: int) -> float:
        """
        Calculates Recall@K.

        :param retrieved_docs: A list of (doc_id, score) tuples, ranked.
        :param qrels: A dictionary of relevant document IDs and their relevance scores for a query.
        :param k: The number of top documents to consider.
        :return: Recall@K value.
        """
        if not qrels: # No relevant documents for this query
            return 0.0
        
        num_relevant_docs = sum(1 for score in qrels.values() if score > 0)
        if num_relevant_docs == 0: # Should not happen if not qrels is checked, but as a safeguard
            return 0.0

        relevant_retrieved = 0
        for i in range(min(k, len(retrieved_docs))):
            doc_id = retrieved_docs[i][0]
            if doc_id in qrels and qrels[doc_id] > 0:
                relevant_retrieved += 1
        return relevant_retrieved / num_relevant_docs

    @staticmethod
    def calculate_ndcg_at_k(retrieved_docs: List[Tuple[str, float]], qrels: Dict[str, int], k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG@K).

        :param retrieved_docs: A list of (doc_id, score) tuples, ranked.
        :param qrels: A dictionary of relevant document IDs and their relevance scores for a query.
        :param k: The number of top documents to consider.
        :return: NDCG@K value.
        """
        if not retrieved_docs or k <= 0:
            return 0.0

        dcg = 0.0
        for i in range(min(k, len(retrieved_docs))):
            doc_id = retrieved_docs[i][0]
            relevance = qrels.get(doc_id, 0)
            dcg += relevance / math.log2(i + 2) # i+1 for 1-based indexing, +1 for log argument

        # Calculate Ideal DCG (IDCG)
        ideal_relevance_scores = sorted([score for score in qrels.values() if score > 0], reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_relevance_scores))):
            idcg += ideal_relevance_scores[i] / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_average_precision(retrieved_docs: List[Tuple[str, float]], qrels: Dict[str, int]) -> float:
        """
        Calculates Average Precision (AP).

        :param retrieved_docs: A list of (doc_id, score) tuples, ranked.
        :param qrels: A dictionary of relevant document IDs and their relevance scores for a query.
        :return: Average Precision value.
        """
        if not qrels:
            return 0.0

        relevant_docs_in_qrels = {doc_id for doc_id, score in qrels.items() if score > 0}
        
        if not relevant_docs_in_qrels: # No relevant documents for this query
            return 0.0

        sum_precisions = 0.0
        relevant_count = 0
        
        for i, (doc_id, _) in enumerate(retrieved_docs):
            if doc_id in relevant_docs_in_qrels:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant_docs_in_qrels)

def evaluate_models(models_to_evaluate: Dict[str, Union[VectorSpaceModel, BERTRetrievalModel, HybridRanker]],
                    queries: Dict[str, Query], qrels: Dict[str, Dict[str, int]],
                    top_k_values: List[int]):
    """
    Evaluates multiple retrieval models and prints their performance metrics.

    :param models_to_evaluate: A dictionary of model instances (e.g., {"TF-IDF": vsm_model}).
    :param queries: A dictionary of Query objects, keyed by query_id.
    :param qrels: A dictionary of relevance judgments {query_id: {doc_id: relevance_score}}.
    :param top_k_values: A list of K values for P@K, R@K, NDCG@K.
    """
    evaluator = Evaluator()
    all_results = defaultdict(lambda: defaultdict(float))
    
    max_k_for_search = max(top_k_values) # Get max K to perform single search per query

    for model_name, model_obj in models_to_evaluate.items():
        print(f"Evaluating {model_name}...")
        
        # Define the search function dynamically based on the model type and name
        if model_name == "TF-IDF":
            search_func = lambda q_obj, top_k: model_obj.search(q_obj.text, top_k=top_k)
        elif model_name == "BERT":
            search_func = lambda q_obj, top_k: model_obj.search(q_obj.text, top_k=top_k)
        elif model_name == "Hybrid":
            # Corrected call for HybridRanker's hybrid_search method
            # Removed 'fusion_method' and added specific weights and top_k_bert_initial
            search_func = lambda q_obj, top_k: model_obj.hybrid_search(
                q_obj.text, 
                top_k=top_k, 
                vsm_weight=0.1,               # Matching the weights set in main.py
                bert_weight=0.9,              # Matching the weights set in main.py
                top_k_bert_initial=200        # Matching the initial K set in main.py
            )
        else:
            raise ValueError(f"Unknown model name for evaluation: {model_name}")

        query_ids = list(queries.keys())
        
        # For each query, perform search and calculate metrics
        # Use tqdm for a progress bar
        for query_id in tqdm(query_ids, desc=f"Running queries for {model_name}"):
            query = queries[query_id]
            query_qrels = qrels.get(query_id, {})

            # Perform search for the current query
            # We search for max_k_for_search to cover all P@K, R@K, NDCG@K values
            results = search_func(query, top_k=max_k_for_search) 
            
            # Calculate metrics for current query
            # Precision@K
            for k in top_k_values:
                p_at_k = evaluator.calculate_precision_at_k(results, query_qrels, k)
                all_results[model_name][f'P@{k}'] += p_at_k

            # Recall@K
            for k in top_k_values:
                r_at_k = evaluator.calculate_recall_at_k(results, query_qrels, k)
                all_results[model_name][f'R@{k}'] += r_at_k

            # NDCG@K
            for k in top_k_values:
                ndcg_at_k = evaluator.calculate_ndcg_at_k(results, query_qrels, k)
                all_results[model_name][f'NDCG@{k}'] += ndcg_at_k

            # MAP (Mean Average Precision)
            ap = evaluator.calculate_average_precision(results, query_qrels)
            all_results[model_name]['MAP'] += ap

        # Average metrics across all queries
        num_queries = len(query_ids)
        for metric, total_value in all_results[model_name].items():
            all_results[model_name][metric] = total_value / num_queries

    # Print summary table
    print("\n--- Evaluation Summary ---")
    headers = ["Model"] + [f"P@{k}" for k in top_k_values] + \
              [f"R@{k}" for k in top_k_values] + \
              [f"NDCG@{k}" for k in top_k_values] + ["MAP"]
    
    # Dynamically determine column widths based on max header/value length
    col_widths = {header: len(header) for header in headers}
    for model_name in models_to_evaluate.keys():
        col_widths["Model"] = max(col_widths["Model"], len(model_name))
        for k in top_k_values:
            col_widths[f"P@{k}"] = max(col_widths[f"P@{k}"], len(f"{all_results[model_name][f'P@{k}']:.4f}"))
            col_widths[f"R@{k}"] = max(col_widths[f"R@{k}"], len(f"{all_results[model_name][f'R@{k}']:.4f}"))
            col_widths[f"NDCG@{k}"] = max(col_widths[f"NDCG@{k}"], len(f"{all_results[model_name][f'NDCG@{k}']:.4f}"))
        col_widths["MAP"] = max(col_widths["MAP"], len(f"{all_results[model_name]['MAP']:.4f}"))

    # Print header row
    header_str = ""
    for header in headers:
        header_str += f"{header:<{col_widths[header] + 2}}" # Add 2 for padding
    print(header_str)
    print("-" * len(header_str))

    # Print results for each model
    for model_name, metrics in all_results.items():
        row_str = f"{model_name:<{col_widths['Model'] + 2}}"
        for k in top_k_values:
            row_str += f"{metrics[f'P@{k}']:{col_widths[f'P@{k}']}.4f}  "
        for k in top_k_values:
            row_str += f"{metrics[f'R@{k}']:{col_widths[f'R@{k}']}.4f}  "
        for k in top_k_values:
            row_str += f"{metrics[f'NDCG@{k}']:{col_widths[f'NDCG@{k}']}.4f}  "
        row_str += f"{metrics['MAP']:{col_widths['MAP']}.4f}  "
        print(row_str)