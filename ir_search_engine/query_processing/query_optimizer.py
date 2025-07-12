import math
from collections import Counter
from typing import List, Dict, Tuple, Union

from ir_search_engine.data_processing import TextPreprocessor
from ir_search_engine.retrieval_models import VectorSpaceModel, BERTRetrievalModel

class QueryOptimizer:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor

    def expand_query_with_prf(self, 
                              original_query: str, 
                              retrieval_model: Union[VectorSpaceModel, BERTRetrievalModel], 
                              raw_documents_dict: Dict[Union[int, str], str],
                              top_n_docs_for_prf: int = 5, 
                              num_expansion_terms: int = 3) -> str:
        print(f"Running Query Optimization (PRF)...")
        print(f"Original query: '{original_query}'")
        
        corrected_query = self.preprocessor.correct_query_spelling(original_query)
        if corrected_query.lower() != original_query.lower():
            print(f"Spell correction applied: '{original_query}' -> '{corrected_query}'")
        else:
            print(f"Spell correction: No changes needed for '{original_query}'")
        
        print(f"Using {retrieval_model.__class__.__name__} for initial search...")

        initial_results = retrieval_model.search(corrected_query, top_k=top_n_docs_for_prf)

        if not initial_results:
            print("No documents found for pseudo-relevance feedback. Query not expanded.")
            return corrected_query

        feedback_doc_texts: List[str] = []
        for doc_id, _score in initial_results:
            doc_text = raw_documents_dict.get(doc_id)
            if doc_text:
                feedback_doc_texts.append(doc_text)
            else:
                if isinstance(doc_id, str) and doc_id.isdigit():
                    doc_text_fallback = raw_documents_dict.get(int(doc_id))
                    if doc_text_fallback: feedback_doc_texts.append(doc_text_fallback)
                elif isinstance(doc_id, int):
                    doc_text_fallback = raw_documents_dict.get(str(doc_id))
                    if doc_text_fallback: feedback_doc_texts.append(doc_text_fallback)
                
                if not doc_text and not doc_text_fallback:
                    print(f"Warning: Could not find raw text for document ID {doc_id} for PRF.")

        if not feedback_doc_texts:
            print("Could not retrieve text for any feedback documents. Query not expanded.")
            return corrected_query

        all_feedback_terms: List[str] = []
        for text in feedback_doc_texts:
            preprocessed_tokens_list = self.preprocessor.preprocess_query(
                text, use_stemming=False, use_lemmatization=True, add_ngrams=False
            )
            all_feedback_terms.extend(preprocessed_tokens_list)

        term_counts = Counter(all_feedback_terms)

        preprocessed_corrected_query_terms = set(
            self.preprocessor.preprocess_query(
                corrected_query, use_stemming=False, use_lemmatization=True, add_ngrams=False
            )
        )
        
        candidate_expansion_terms_ranked = []
        for term, count in term_counts.most_common():
            if (term not in preprocessed_corrected_query_terms and 
                term not in self.preprocessor.stop_words and
                len(term) > 1):
                candidate_expansion_terms_ranked.append(term)
            
            if len(candidate_expansion_terms_ranked) >= num_expansion_terms:
                break

        expanded_terms = candidate_expansion_terms_ranked[:num_expansion_terms]

        if expanded_terms:
            expanded_query = f"{corrected_query} {' '.join(expanded_terms)}"
            print(f"Final expanded query: '{expanded_query}' (added: {', '.join(expanded_terms)})")
            return expanded_query
        else:
            print("No suitable expansion terms found after filtering. Query not expanded.")
            return corrected_query