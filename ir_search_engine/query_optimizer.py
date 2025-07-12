# ir_search_engine/query_optimizer.py

import math
from collections import Counter
from typing import List, Dict, Tuple, Union

# Import modules from your project
from preprocessing import TextPreprocessor
from retrieval_model import VectorSpaceModel
from bert_retrieval import BERTRetrievalModel

class QueryOptimizer:
    """
    Provides methods for optimizing user queries, e.g., through pseudo-relevance feedback (PRF).
    """
    def __init__(self, preprocessor: TextPreprocessor):
        """
        Initializes the QueryOptimizer.
        :param preprocessor: An instance of TextPreprocessor.
        """
        self.preprocessor = preprocessor

    def expand_query_with_prf(self, 
                              original_query: str, 
                              retrieval_model: Union[VectorSpaceModel, BERTRetrievalModel], 
                              raw_documents_dict: Dict[Union[int, str], str], # Added for access to full text
                              top_n_docs_for_prf: int = 5, 
                              num_expansion_terms: int = 3) -> str:
        """
        Expands the original query using Pseudo-Relevance Feedback (PRF).
        Automatically applies spell correction before PRF expansion.

        Performs spell correction on the original query, then performs an initial search,
        assumes the top_n_docs_for_prf are relevant, extracts the most significant terms
        from these documents, and adds them to the corrected query.

        :param original_query: The user's initial query string.
        :param retrieval_model: The specific model instance (VSM or BERT) to use for initial search.
        :param raw_documents_dict: A dictionary of {doc_id: raw_text} for retrieving document content.
        :param top_n_docs_for_prf: The number of top documents to consider for feedback.
        :param num_expansion_terms: The number of top terms to add to the query.
        :return: The expanded query string with spell correction applied.
        """
        print(f"\n--- Running Query Optimization (PRF) ---")
        print(f"Original query: '{original_query}'")
        
        # Step 1: Apply spell correction to the original query
        corrected_query = self.preprocessor.correct_query_spelling(original_query)
        if corrected_query.lower() != original_query.lower():
            print(f"Spell correction applied: '{original_query}' -> '{corrected_query}'")
        else:
            print(f"Spell correction: No changes needed for '{original_query}'")
        
        print(f"Using {retrieval_model.__class__.__name__} for initial search...")

        # 2. Perform initial search using the corrected query
        initial_results = retrieval_model.search(corrected_query, top_k=top_n_docs_for_prf)

        if not initial_results:
            print("No documents found for pseudo-relevance feedback. Query not expanded.")
            return corrected_query  # Return corrected query even if no expansion

        # 3. Get the original text content of the feedback documents
        feedback_doc_texts: List[str] = []
        for doc_id, _score in initial_results:
            doc_text = raw_documents_dict.get(doc_id)
            if doc_text:
                feedback_doc_texts.append(doc_text)
            else:
                # Fallback for potential ID type mismatch (int vs str) if needed, though raw_documents_dict should handle it
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
            return corrected_query  # Return corrected query even if no expansion

        # 4. Preprocess feedback documents and extract candidate terms
        all_feedback_terms: List[str] = []
        for text in feedback_doc_texts:
            # Use the preprocessor for consistent tokenization and normalization
            # Using same settings as VSM indexing (lemmatization, no n-grams for base terms)
            preprocessed_tokens_list = self.preprocessor.preprocess_query(
                text, use_stemming=False, use_lemmatization=True, add_ngrams=False
            )
            all_feedback_terms.extend(preprocessed_tokens_list) # assuming space-separated terms

        # 5. Filter and rank candidate expansion terms
        term_counts = Counter(all_feedback_terms)

        # Get preprocessed terms from the corrected query to avoid adding them back
        preprocessed_corrected_query_terms = set(
            self.preprocessor.preprocess_query(
                corrected_query, use_stemming=False, use_lemmatization=True, add_ngrams=False
            )
        )
        
        # Filter out corrected query terms, stopwords, and very short terms
        candidate_expansion_terms_ranked = []
        # Iterate through most common terms from feedback documents
        for term, count in term_counts.most_common():
            if (term not in preprocessed_corrected_query_terms and 
                term not in self.preprocessor.stop_words and # Fixed: use stop_words instead of stopwords
                len(term) > 1): # Exclude single-character terms, usually noise
                candidate_expansion_terms_ranked.append(term)
            
            if len(candidate_expansion_terms_ranked) >= num_expansion_terms:
                break # Stop once we have enough terms

        # Select the top N terms
        expanded_terms = candidate_expansion_terms_ranked[:num_expansion_terms]

        if expanded_terms:
            expanded_query = f"{corrected_query} {' '.join(expanded_terms)}"
            print(f"Final expanded query: '{expanded_query}' (added: {', '.join(expanded_terms)})")
            return expanded_query
        else:
            print("No suitable expansion terms found after filtering. Query not expanded.")
            return corrected_query  # Return corrected query even if no expansion