from collections import defaultdict
import numpy as np

class IREvaluator:
    def __init__(self):
        pass

    def calculate_precision_at_k(self, relevant_docs_for_query, retrieved_docs, k):
        if not retrieved_docs:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        num_relevant_retrieved = len([doc_id for doc_id, _ in retrieved_at_k if doc_id in relevant_docs_for_query])
        return num_relevant_retrieved / k

    def calculate_recall_at_k(self, relevant_docs_for_query, retrieved_docs, k):
        if not relevant_docs_for_query:
            return 1.0 # If there are no relevant documents, recall is perfectly achieved if nothing is retrieved
        
        retrieved_at_k = retrieved_docs[:k]
        num_relevant_retrieved = len([doc_id for doc_id, _ in retrieved_at_k if doc_id in relevant_docs_for_query])
        return num_relevant_retrieved / len(relevant_docs_for_query)

    def calculate_average_precision(self, relevant_docs_for_query, retrieved_docs):
        if not relevant_docs_for_query:
            return 0.0
        
        cumulative_precision = 0.0
        num_relevant_found = 0
        
        for i, (doc_id, _) in enumerate(retrieved_docs):
            if doc_id in relevant_docs_for_query:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                cumulative_precision += precision_at_i
        
        if num_relevant_found == 0:
            return 0.0
        
        return cumulative_precision / len(relevant_docs_for_query)

    def calculate_mrr(self, relevant_docs_for_query, retrieved_docs):
        for i, (doc_id, _) in enumerate(retrieved_docs):
            if doc_id in relevant_docs_for_query:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate(self, qrels, retrieved_results_per_query, k=10):
        """
        Evaluates the IR system.
        Args:
            qrels (list of Qrel): Ground truth relevance judgments.
            retrieved_results_per_query (dict): {query_id: [(doc_id, score), ...]}
            k (int): Cut-off for Precision@K.
        Returns:
            (MAP, Avg_Recall, Avg_P@K, Avg_MRR)
        """
        print(f"Evaluating system with k={k}...")
        
        # Organize qrels by query_id for easy lookup
        relevant_docs_by_query = defaultdict(set)
        for qrel in qrels:
            # Assuming relevance > 0 means relevant
            if qrel.relevance > 0:
                relevant_docs_by_query[qrel.query_id].add(qrel.doc_id)

        all_avg_precisions = []
        all_recalls_at_k = []
        all_precisions_at_k = []
        all_mrrs = []

        num_queries_with_relevant_docs = 0
        num_queries_with_retrievals = 0

        for query_id, retrieved_docs in retrieved_results_per_query.items():
            relevant_docs = relevant_docs_by_query.get(query_id, set())

            # Skip queries with no relevant documents in qrels for AP and MRR calculation
            # For recall, if no relevant docs, and system retrieves nothing, it's 1.0. If it retrieves something, it's 0.0.
            # We will include all queries for overall average calculations unless specifically handling no relevant cases.

            if relevant_docs:
                num_queries_with_relevant_docs += 1
                all_avg_precisions.append(self.calculate_average_precision(relevant_docs, retrieved_docs))
                all_mrrs.append(self.calculate_mrr(relevant_docs, retrieved_docs))
            
            if retrieved_docs: # Only count queries for which results were returned
                num_queries_with_retrievals += 1
                all_precisions_at_k.append(self.calculate_precision_at_k(relevant_docs, retrieved_docs, k))
                all_recalls_at_k.append(self.calculate_recall_at_k(relevant_docs, retrieved_docs, k))
            else: # If no retrieved docs, but there were relevant docs, P@K and R@K are 0
                if relevant_docs:
                    all_precisions_at_k.append(0.0)
                    all_recalls_at_k.append(0.0)

        map_score = np.mean(all_avg_precisions) if all_avg_precisions else 0.0
        avg_recall_at_k = np.mean(all_recalls_at_k) if all_recalls_at_k else 0.0
        avg_precision_at_k = np.mean(all_precisions_at_k) if all_precisions_at_k else 0.0
        mrr_score = np.mean(all_mrrs) if all_mrrs else 0.0

        print(f"MAP: {map_score:.4f}")
        print(f"Recall@{k}: {avg_recall_at_k:.4f}")
        print(f"Precision@{k}: {avg_precision_at_k:.4f}")
        print(f"MRR: {mrr_score:.4f}")

        return map_score, avg_recall_at_k, avg_precision_at_k, mrr_score

# Example usage (will be in main.py)
# from evaluation import IREvaluator
# from data_loader import DataLoader
# # Assume you have qrels and retrieved_results_per_query from your search process
# # qrels: list of Qrel namedtuples
# # retrieved_results_per_query: dict like {query_id: [(doc_id, score), ...]}
#
# evaluator = IREvaluator()
# # Example dummy data:
# # qrels_dummy = [Qrel("Q1", "D1", 1, "0"), Qrel("Q1", "D2", 1, "0"), Qrel("Q2", "D3", 1, "0")]
# # retrieved_dummy = {
# #    "Q1": [("D1", 0.9), ("D3", 0.8), ("D2", 0.7)],
# #    "Q2": [("D4", 0.95), ("D3", 0.85)]
# # }
# # map_score, recall, precision10, mrr = evaluator.evaluate(qrels_dummy, retrieved_dummy, k=2)