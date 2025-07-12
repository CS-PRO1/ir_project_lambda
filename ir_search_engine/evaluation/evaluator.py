import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Any, Optional, Tuple
from tqdm import tqdm

from ir_search_engine.data_processing import Query, Qrel, TextPreprocessor
from ir_search_engine.retrieval_models import VectorSpaceModel, BERTRetrievalModel, HybridRanker
from ir_search_engine.query_processing import QueryOptimizer
from ir_search_engine.clustering import DocumentClusterer

def calculate_precision(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    relevant_retrieved = 0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
            relevant_retrieved += 1
    return relevant_retrieved / k if k > 0 else 0.0

def calculate_recall(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int], k: int) -> float:
    num_relevant_docs = sum(1 for rel in relevant_docs.values() if rel > 0)
    if num_relevant_docs == 0:
        return 1.0
    
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
        dcg += relevance / np.log2(i + 2)
    return dcg

def calculate_idcg(relevant_docs: Dict[Union[int, str], int], k: int) -> float:
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
    precisions = []
    num_relevant_found = 0
    
    true_relevant_doc_ids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    
    if not true_relevant_doc_ids:
        return 0.0
    
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in true_relevant_doc_ids:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / (i + 1)
            precisions.append(precision_at_k)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(true_relevant_doc_ids)

def calculate_map(retrieved_docs_for_query: List[Tuple[Union[int, str], float]], relevant_docs: Dict[Union[int, str], int]) -> float:
    return calculate_ap(retrieved_docs_for_query, relevant_docs)

def calculate_reciprocal_rank(retrieved_doc_ids: List[Union[int, str]], relevant_docs: Dict[Union[int, str], int]) -> float:
    true_relevant_doc_ids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    
    if not true_relevant_doc_ids:
        return 0.0
    
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in true_relevant_doc_ids:
            return 1.0 / (i + 1)
    
    return 0.0

def evaluate_models(
    models: Dict[str, Union[VectorSpaceModel, BERTRetrievalModel, HybridRanker]],
    queries: Dict[Union[int, str], Query],
    qrels: Dict[Union[int, str], Dict[Union[int, str], int]],
    raw_documents_dict: Dict[Union[int, str], str],
    preprocessor: TextPreprocessor,
    query_optimizer: QueryOptimizer,
    k_values: List[int] = [10],
    document_clusterer: Optional[DocumentClusterer] = None,
    use_clustering_for_bert: bool = False,
    use_prf: bool = False,
    prf_initial_model_name: Optional[str] = None,
    prf_top_n_docs: int = 5,
    prf_num_expansion_terms: int = 3,
    prf_final_model_name: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    
    all_results = defaultdict(lambda: defaultdict(list))
    
    for model_name, model_instance in models.items():
        if model_instance is None:
            print(f"Skipping evaluation for {model_name}: model not initialized.")
            continue
        
        print(f"Evaluating {model_name}...")
        for query_id, query_obj in tqdm(queries.items(), desc=f"Evaluating {model_name}"):
            if model_name.lower() == 'hybrid':
                retrieved_docs = model_instance.hybrid_search(
                    query_obj.text, 
                    top_k=max(k_values) * 2,
                    vsm_weight=0.1, 
                    bert_weight=0.9, 
                    top_k_bert_initial=max(k_values) * 5
                )
            else:
                retrieved_docs = model_instance.search(query_obj.text, top_k=max(k_values) * 2)

            all_results[model_name][query_id] = retrieved_docs

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
                
                retrieved_docs_clustered = bert_model_instance.search(
                    query_obj.text, 
                    top_k=max(k_values) * 2,
                    candidate_doc_ids=cluster_doc_ids
                )
                all_results[model_name_clustered][query_id] = retrieved_docs_clustered

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

    final_metrics = {}
    for model_name, query_results in all_results.items():
        if not query_results: 
            continue
        
        map_scores = []
        recall_scores = []
        precision_at_10_scores = []
        rr_scores = []

        for query_id in qrels.keys():
            relevant_docs_for_query = qrels.get(query_id, {})
            retrieved_docs_for_query = query_results.get(query_id, [])

            num_true_relevant = sum(1 for rel in relevant_docs_for_query.values() if rel > 0)
            
            if num_true_relevant == 0:
                if not retrieved_docs_for_query:
                    pass 
                else:
                    precision_at_10_scores.append(0.0)
                    recall_scores.append(1.0)
                    rr_scores.append(0.0)
                map_scores.append(0.0)
                continue

            if not retrieved_docs_for_query:
                precision_at_10_scores.append(0.0)
                recall_scores.append(0.0)
                rr_scores.append(0.0)
                map_scores.append(0.0)
                continue

            retrieved_doc_ids_only = [doc_id for doc_id, _ in retrieved_docs_for_query]

            map_scores.append(calculate_map(retrieved_doc_ids_only, relevant_docs_for_query))
            recall_scores.append(calculate_recall(retrieved_doc_ids_only, relevant_docs_for_query, len(retrieved_doc_ids_only)))
            precision_at_10_scores.append(calculate_precision(retrieved_doc_ids_only, relevant_docs_for_query, 10))
            rr_scores.append(calculate_reciprocal_rank(retrieved_doc_ids_only, relevant_docs_for_query))

        avg_map = np.mean(map_scores) if map_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        avg_precision_at_10 = np.mean(precision_at_10_scores) if precision_at_10_scores else 0.0
        avg_rr = np.mean(rr_scores) if rr_scores else 0.0

        final_metrics[model_name] = {
            'map': avg_map,
            'recall': avg_recall,
            'precision_at_10': avg_precision_at_10,
            'mrr': avg_rr
        }
    
    return final_metrics