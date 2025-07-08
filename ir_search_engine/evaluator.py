import math
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union
from tqdm import tqdm

# Assuming Qrel is defined as a namedtuple in data_loader.py
# If not, you might need to import it or define its expected structure.
# from data_loader import Qrel # Uncomment if Qrel is a class/object from data_loader

# Helper type for preprocessed qrels: {query_id: {doc_id: relevance_score}}
RelevantDocsInfo = Dict[Union[int, str], Dict[Union[int, str], int]]

def _prepare_qrels_for_eval(qrels: List) -> RelevantDocsInfo:
    """
    Transforms a list of Qrel objects (from data_loader) into a more accessible
    dictionary format for evaluation: {query_id: {doc_id: relevance_score}}.
    """
    prepared_qrels = defaultdict(dict)
    for qrel in qrels:
        # Ensure doc_id is consistent type (str or int) as in results
        # Assuming qrel.doc_id is already the correct type,
        # otherwise convert it here, e.g., str(qrel.doc_id) or int(qrel.doc_id)
        prepared_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    return prepared_qrels

def precision_at_k(retrieved_docs: List[Tuple[Union[int, str], float]], 
                   relevant_docs_for_query: Dict[Union[int, str], int], 
                   k: int) -> float:
    """
    Calculates Precision@k for a single query.

    :param retrieved_docs: A list of (doc_id, score) tuples, ordered by rank.
    :param relevant_docs_for_query: A dict of {doc_id: relevance_score} for this query.
                                    Relevance score > 0 means relevant.
    :param k: The number of top documents to consider.
    :return: Precision@k score.
    """
    if not retrieved_docs or k == 0:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    num_relevant_in_k = 0
    for doc_id, _ in retrieved_k:
        # Check if the document is relevant (relevance score > 0)
        if relevant_docs_for_query.get(doc_id, 0) > 0:
            num_relevant_in_k += 1
            
    return num_relevant_in_k / k

def recall_at_k(retrieved_docs: List[Tuple[Union[int, str], float]], 
                relevant_docs_for_query: Dict[Union[int, str], int], 
                k: int) -> float:
    """
    Calculates Recall@k for a single query.

    :param retrieved_docs: A list of (doc_id, score) tuples, ordered by rank.
    :param relevant_docs_for_query: A dict of {doc_id: relevance_score} for this query.
                                    Relevance score > 0 means relevant.
    :param k: The number of top documents to consider.
    :return: Recall@k score.
    """
    total_relevant_docs = sum(1 for score in relevant_docs_for_query.values() if score > 0)
    if total_relevant_docs == 0:
        return 0.0 # Cannot calculate recall if no relevant documents exist

    retrieved_k = retrieved_docs[:k]
    num_relevant_in_k = 0
    for doc_id, _ in retrieved_k:
        if relevant_docs_for_query.get(doc_id, 0) > 0:
            num_relevant_in_k += 1
            
    return num_relevant_in_k / total_relevant_docs

def average_precision(retrieved_docs: List[Tuple[Union[int, str], float]], 
                       relevant_docs_for_query: Dict[Union[int, str], int]) -> float:
    """
    Calculates Average Precision (AP) for a single query.

    :param retrieved_docs: A list of (doc_id, score) tuples, ordered by rank.
    :param relevant_docs_for_query: A dict of {doc_id: relevance_score} for this query.
                                    Relevance score > 0 means relevant.
    :return: Average Precision score.
    """
    if not retrieved_docs:
        return 0.0

    total_relevant_docs = sum(1 for score in relevant_docs_for_query.values() if score > 0)
    if total_relevant_docs == 0:
        return 0.0 # No relevant documents for this query, so AP is 0

    sum_precisions = 0.0
    num_relevant_found = 0
    for i, (doc_id, _) in enumerate(retrieved_docs):
        if relevant_docs_for_query.get(doc_id, 0) > 0:
            num_relevant_found += 1
            precision_at_current_rank = num_relevant_found / (i + 1)
            sum_precisions += precision_at_current_rank
            
    return sum_precisions / total_relevant_docs if total_relevant_docs > 0 else 0.0

def mean_average_precision(query_results: Dict[Union[int, str], List[Tuple[Union[int, str], float]]], 
                           prepared_qrels: RelevantDocsInfo) -> float:
    """
    Calculates Mean Average Precision (MAP) across multiple queries.

    :param query_results: A dict of {query_id: [(doc_id, score), ...]} for all queries.
    :param prepared_qrels: Preprocessed qrels {query_id: {doc_id: relevance_score}}.
    :return: MAP score.
    """
    if not query_results:
        return 0.0

    total_aps = 0.0
    num_queries_with_relevant_docs = 0

    for query_id, results in query_results.items():
        relevant_docs_for_query = prepared_qrels.get(query_id, {})
        
        # Only include queries that have at least one relevant document in the qrels
        # and for which there are relevant documents in the ground truth
        if sum(1 for score in relevant_docs_for_query.values() if score > 0) > 0:
            ap = average_precision(results, relevant_docs_for_query)
            total_aps += ap
            num_queries_with_relevant_docs += 1
            
    return total_aps / num_queries_with_relevant_docs if num_queries_with_relevant_docs > 0 else 0.0


def dcg_at_k(retrieved_docs: List[Tuple[Union[int, str], float]], 
             relevant_docs_for_query: Dict[Union[int, str], int], 
             k: int) -> float:
    """
    Calculates Discounted Cumulative Gain (DCG@k) for a single query.
    Assumes relevance scores are integers (e.g., 0, 1, 2, 3).

    :param retrieved_docs: A list of (doc_id, score) tuples, ordered by rank.
    :param relevant_docs_for_query: A dict of {doc_id: relevance_score} for this query.
    :param k: The number of top documents to consider.
    :return: DCG@k score.
    """
    dcg = 0.0
    for i, (doc_id, _) in enumerate(retrieved_docs[:k]):
        relevance = relevant_docs_for_query.get(doc_id, 0)
        # log base 2 of (rank + 1)
        # i is 0-indexed, so rank is i+1. Denominator is log2(rank+1) = log2(i+2)
        dcg += relevance / math.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved_docs: List[Tuple[Union[int, str], float]], 
              relevant_docs_for_query: Dict[Union[int, str], int], 
              k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@k) for a single query.

    :param retrieved_docs: A list of (doc_id, score) tuples, ordered by rank.
    :param relevant_docs_for_query: A dict of {doc_id: relevance_score} for this query.
    :param k: The number of top documents to consider.
    :return: NDCG@k score.
    """
    dcg = dcg_at_k(retrieved_docs, relevant_docs_for_query, k)

    # Calculate Ideal DCG (IDCG)
    # Get all relevant docs for this query and sort by relevance score descending
    ideal_relevance_scores = sorted([
        score for score in relevant_docs_for_query.values() if score > 0
    ], reverse=True)
    
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevance_scores[:k]):
        idcg += relevance / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0 # Cannot normalize if IDCG is 0 (no relevant docs for this query)
    
    return dcg / idcg

def evaluate_models(models: Dict[str, object], # Changed type hint to simply 'object'
                    queries: List, 
                    qrels: List, 
                    top_k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[str, float]]:
    """
    Evaluates multiple retrieval models and returns their scores for various metrics.

    :param models: A dictionary mapping model names (str) to model objects (VSM, BERT, HybridRanker).
    :param queries: A list of Query objects.
    :param qrels: A list of Qrel objects.
    :param top_k_values: A list of k values for metrics like Precision@k, Recall@k, NDCG@k.
    :return: A dictionary of {model_name: {metric_name: score}}.
    """
    print("\n--- Running Model Evaluation ---")
    
    prepared_qrels = _prepare_qrels_for_eval(qrels)
    
    evaluation_results = defaultdict(lambda: defaultdict(float))

    for model_name, model_obj in models.items():
        print(f"Evaluating {model_name}...")
        query_results_for_model = {}
        
        # Determine how to call the search method based on model type
        search_func = None
        if hasattr(model_obj, 'hybrid_search'): # If it's the HybridRanker instance
            search_func = lambda q_text, top_k: model_obj.hybrid_search(q_text, top_k=top_k, fusion_method='rrf')
        elif hasattr(model_obj, 'search'): # If it's VSM or BERT model
            search_func = lambda q_text, top_k: model_obj.search(q_text, top_k=top_k)
        else:
            print(f"Error: Model {model_name} does not have a recognized search function. Skipping.")
            continue

        for query in tqdm(queries, desc=f"Running queries for {model_name}"):
            # We must pass query.query_text, not the whole query object
            results = search_func(query.query_text, top_k=max(top_k_values)) # Search for max k to cover all metrics
            query_results_for_model[query.query_id] = results
        
        # Calculate metrics for the current model
        num_evaluated_queries = 0
        total_ap = 0.0
        total_ndcg_at_k = defaultdict(float)
        total_precision_at_k = defaultdict(float)
        total_recall_at_k = defaultdict(float)

        for query_id, results in query_results_for_model.items():
            relevant_docs_for_query = prepared_qrels.get(query_id, {})
            # Only evaluate queries that actually have relevant documents in qrels
            if sum(1 for score in relevant_docs_for_query.values() if score > 0) == 0:
                continue
            
            num_evaluated_queries += 1
            total_ap += average_precision(results, relevant_docs_for_query)

            for k in top_k_values:
                total_ndcg_at_k[k] += ndcg_at_k(results, relevant_docs_for_query, k)
                total_precision_at_k[k] += precision_at_k(results, relevant_docs_for_query, k)
                total_recall_at_k[k] += recall_at_k(results, relevant_docs_for_query, k)

        if num_evaluated_queries > 0:
            evaluation_results[model_name]['MAP'] = total_ap / num_evaluated_queries
            for k in top_k_values:
                evaluation_results[model_name][f'NDCG@{k}'] = total_ndcg_at_k[k] / num_evaluated_queries
                evaluation_results[model_name][f'P@{k}'] = total_precision_at_k[k] / num_evaluated_queries
                evaluation_results[model_name][f'R@{k}'] = total_recall_at_k[k] / num_evaluated_queries
        else:
            print(f"No queries with relevant documents in qrels were found for {model_name} to evaluate.")

    print("\n--- Evaluation Summary ---")
    # Print a nicely formatted table
    header = ["Model"] + [f"P@{k}" for k in top_k_values] + \
             [f"R@{k}" for k in top_k_values] + \
             [f"NDCG@{k}" for k in top_k_values] + \
             ["MAP"]
    
    # Sort models for consistent display
    sorted_model_names = sorted(evaluation_results.keys())

    # Build rows
    data_rows = []
    for model_name in sorted_model_names:
        row = [model_name]
        for k in top_k_values:
            row.append(f"{evaluation_results[model_name].get(f'P@{k}', 0.0):.4f}")
        for k in top_k_values:
            row.append(f"{evaluation_results[model_name].get(f'R@{k}', 0.0):.4f}")
        for k in top_k_values:
            row.append(f"{evaluation_results[model_name].get(f'NDCG@{k}', 0.0):.4f}")
        row.append(f"{evaluation_results[model_name].get('MAP', 0.0):.4f}")
        data_rows.append(row)

    # Determine column widths for pretty printing
    column_widths = [max(len(str(item)) for item in col) for col in zip(header, *data_rows)]
    
    # Print header
    print(" | ".join(header[i].ljust(column_widths[i]) for i in range(len(header))))
    print("-|-".join('-' * column_widths[i] for i in range(len(header))))

    # Print data rows
    for row in data_rows:
        print(" | ".join(row[i].ljust(column_widths[i]) for i in range(len(row))))
    
    return evaluation_results


# Example Usage (for direct testing of evaluator.py)
if __name__ == "__main__":
    print("--- Testing Evaluator (Standalone) ---")

    # Dummy Qrel namedtuple for testing if data_loader is not available
    from collections import namedtuple
    TestQrel = namedtuple('Qrel', ['query_id', 'doc_id', 'relevance'])
    TestQuery = namedtuple('Query', ['query_id', 'query_text'])

    # Sample Qrels: query_id, doc_id, relevance
    qrels_data_list = [
        TestQrel(query_id='Q1', doc_id='D1', relevance=1),
        TestQrel(query_id='Q1', doc_id='D2', relevance=0), # Not relevant (relevance 0)
        TestQrel(query_id='Q1', doc_id='D3', relevance=1),
        TestQrel(query_id='Q1', doc_id='D4', relevance=2), # Highly relevant (relevance 2)
        TestQrel(query_id='Q2', doc_id='D5', relevance=1),
        TestQrel(query_id='Q2', doc_id='D6', relevance=1),
        TestQrel(query_id='Q3', doc_id='D7', relevance=1), # Query with only one relevant doc
    ]

    # Preprocess qrels for easy lookup
    prepared_qrels = _prepare_qrels_for_eval(qrels_data_list)
    print("\nPrepared Qrels:", prepared_qrels)

    # Sample Retrieved Results (doc_id, score)
    # Query Q1 results
    results_q1_model_A = [('D4', 0.9), ('D1', 0.8), ('D5', 0.7), ('D3', 0.6), ('D6', 0.5)] # D4, D1, D3 are relevant
    results_q1_model_B = [('D1', 0.9), ('D5', 0.8), ('D4', 0.7), ('D2', 0.6), ('D3', 0.5)] # D1, D4, D3 are relevant

    # Query Q2 results
    results_q2_model_A = [('D5', 0.9), ('D7', 0.8), ('D6', 0.7)] # D5, D6 are relevant (Q2 qrels)
    results_q2_model_B = [('D8', 0.9), ('D5', 0.8), ('D9', 0.7)] # D5 is relevant (Q2 qrels)

    # Query Q3 results (for model A, no relevant documents retrieved)
    results_q3_model_A = [('D8', 0.9), ('D9', 0.8), ('D10', 0.7)]
    results_q3_model_B = [('D7', 0.9), ('D11', 0.8), ('D12', 0.7)] # D7 is relevant (Q3 qrels)


    # Combine results for evaluate_models function
    # These are mock objects for testing, your actual main.py will pass the real model instances
    mock_vsm = type('MockVSM', (), {'search': lambda s, q_text, top_k: {
        'Q1': results_q1_model_A,
        'Q2': results_q2_model_A,
        'Q3': results_q3_model_A
    }.get(q_text, [])[:top_k]})()

    mock_bert = type('MockBERT', (), {'search': lambda s, q_text, top_k: {
        'Q1': results_q1_model_B,
        'Q2': results_q2_model_B,
        'Q3': results_q3_model_B
    }.get(q_text, [])[:top_k]})()
    
    # Mock a hybrid model that uses its own search method
    mock_hybrid_ranker = type('MockHybridRanker', (), {'hybrid_search': lambda s, q_text, top_k, fusion_method: {
        'Q1': [('D4', 0.95), ('D1', 0.9), ('D3', 0.8)], # Example better hybrid results for Q1
        'Q2': [('D5', 0.9), ('D6', 0.8), ('D7', 0.7)],
        'Q3': [('D7', 0.99), ('D10', 0.5), ('D11', 0.4)]
    }.get(q_text, [])[:top_k]})()

    models_to_eval = {
        'Mock VSM': mock_vsm,
        'Mock BERT': mock_bert,
        'Mock Hybrid': mock_hybrid_ranker 
    }

    test_queries = [
        TestQuery('Q1', 'Query 1 Text'),
        TestQuery('Q2', 'Query 2 Text'),
        TestQuery('Q3', 'Query 3 Text')
    ]

    print("\n--- Running evaluate_models test ---")
    evaluation_results = evaluate_models(models_to_eval, test_queries, qrels_data_list, top_k_values=[1, 3, 5])
    print("\nFull Evaluation Results:", evaluation_results)