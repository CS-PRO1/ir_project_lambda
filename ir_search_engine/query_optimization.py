from collections import Counter
import random
from difflib import get_close_matches

class QueryOptimizer:
    def __init__(self, inverted_index, vsm_model, bert_model, clusterer=None, top_k_terms=5):
        self.inverted_index = inverted_index
        self.vsm_model = vsm_model
        self.bert_model = bert_model
        self.clusterer = clusterer # Optional, for cluster-based query expansion
        self.top_k_terms = top_k_terms
        # Build vocabulary from inverted index for spelling correction
        self.vocabulary = self._build_vocabulary()

    def _build_vocabulary(self):
        """Build vocabulary from the inverted index for spelling correction."""
        vocabulary = set()
        if hasattr(self.inverted_index, 'index'):
            vocabulary.update(self.inverted_index.index.keys())
        return vocabulary

    def correct_spelling(self, query_tokens, confidence_threshold=0.6):
        """
        Corrects misspelled words in the query using the vocabulary from the inverted index.
        Args:
            query_tokens (list): List of query tokens
            confidence_threshold (float): Minimum similarity score to consider a correction
        Returns:
            list: Query tokens with corrected spellings
        """
        print(f"Correcting spelling for query: {' '.join(query_tokens)}")
        
        if not self.vocabulary:
            print("No vocabulary available for spelling correction.")
            return query_tokens
        
        corrected_tokens = []
        corrections_made = []
        
        for token in query_tokens:
            # If token exists in vocabulary, keep it as is
            if token in self.vocabulary:
                corrected_tokens.append(token)
                continue
            
            # Find the closest match in vocabulary
            matches = get_close_matches(token, self.vocabulary, n=1, cutoff=confidence_threshold)
            
            if matches:
                corrected_token = matches[0]
                corrected_tokens.append(corrected_token)
                corrections_made.append((token, corrected_token))
                print(f"Corrected '{token}' to '{corrected_token}'")
            else:
                # If no good match found, keep the original token
                corrected_tokens.append(token)
                print(f"No correction found for '{token}', keeping original")
        
        if corrections_made:
            print(f"Spelling corrections made: {corrections_made}")
        else:
            print("No spelling corrections needed.")
            
        return corrected_tokens

    def spelling_correction_with_expansion(self, query_tokens, top_n_docs=5, confidence_threshold=0.6):
        """
        First corrects spelling, then expands the corrected query using TF-IDF.
        This is a more practical approach that combines spelling correction with expansion.
        """
        print(f"Applying spelling correction with expansion for query: {' '.join(query_tokens)}")
        
        # First, correct spelling
        corrected_tokens = self.correct_spelling(query_tokens, confidence_threshold)
        
        # Then expand the corrected query
        expanded_tokens = self.query_expansion_tfidf(corrected_tokens, top_n_docs)
        
        return expanded_tokens

    def query_expansion_tfidf(self, query_tokens, top_n_docs=5):
        """
        Expands query using terms from top-ranked documents based on VSM TF-IDF.
        This is a conceptual expansion. In practice, you'd perform an initial retrieval,
        then analyze the top documents.
        """
        print(f"Expanding query '{' '.join(query_tokens)}' using TF-IDF...")
        # Simulate initial retrieval to get top N documents
        # In a real system, you'd call a search function here.
        # For simplicity, let's just pick some "relevant" docs by checking index.
        candidate_doc_ids = set()
        for term in query_tokens:
            for doc_id, _ in self.inverted_index.get_postings(term):
                candidate_doc_ids.add(doc_id)
                if len(candidate_doc_ids) >= top_n_docs * 2: # Get more than top_n_docs
                    break
            if len(candidate_doc_ids) >= top_n_docs * 2:
                break

        if not candidate_doc_ids:
            print("No candidate documents found for TF-IDF expansion.")
            return query_tokens

        # Get actual top N documents based on VSM score (conceptual retrieval)
        # This requires calculating scores for all candidate docs against the query
        query_vec = self.vsm_model.create_query_vector(query_tokens)
        doc_vectors, doc_ids_all = self.vsm_model.get_document_vectors()
        
        doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids_all)}
        
        # Filter doc_vectors to only include candidate_doc_ids
        relevant_indices = [doc_id_to_idx[did] for did in candidate_doc_ids if did in doc_id_to_idx]
        
        if not relevant_indices:
             print("No relevant documents for TF-IDF expansion after filtering.")
             return query_tokens
             
        candidate_doc_vectors = doc_vectors[relevant_indices]
        candidate_doc_ids_filtered = [doc_ids_all[i] for i in relevant_indices]

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, candidate_doc_vectors).flatten()
        
        ranked_candidates = sorted(zip(candidate_doc_ids_filtered, scores), key=lambda x: x[1], reverse=True)[:top_n_docs]

        expanded_terms = []
        for doc_id, _ in ranked_candidates:
            # Retrieve the original processed tokens for the document from the preprocessed corpus
            # This would require the main application to pass the processed corpus or have index store it.
            # For this conceptual implementation, let's assume `inverted_index` can return original tokens.
            # (In reality, inverted index stores (doc_id, term_freq), not original token list.
            # You'd need access to `processed_corpus` from `preprocessing` module)
            
            # --- For this example, let's simulate getting top terms from some dummy docs ---
            # In a real scenario, you'd iterate through the actual tokens of the top documents
            # and identify terms that are highly weighted (e.g., high TF-IDF in that doc).
            
            # A simple approach: add top terms by frequency from top docs
            # This would require `self.inverted_index` to have a way to get all terms for a doc.
            # Or, `processed_corpus` must be accessible here.
            # Let's assume we have access to processed_corpus for this method for now.
            if hasattr(self.inverted_index, 'processed_corpus_ref'): # A hacky way to get processed_corpus
                doc_tokens = self.inverted_index.processed_corpus_ref.get(doc_id, [])
                term_freqs = Counter(doc_tokens)
                most_common = [term for term, _ in term_freqs.most_common(self.top_k_terms) if term not in query_tokens]
                expanded_terms.extend(most_common)
            else:
                print("Warning: processed_corpus_ref not available in inverted_index for TF-IDF expansion.")

        # Filter out duplicates and original query terms
        new_terms = [term for term in expanded_terms if term not in query_tokens]
        return list(set(query_tokens + new_terms)) # Return unique terms

    def query_expansion_bert_semantic(self, query_embedding, raw_query_text, top_n_docs=5):
        """
        Expands query using semantically similar terms from top-ranked documents
        based on BERT embeddings.
        """
        print(f"Expanding query '{raw_query_text}' using BERT semantic similarity...")
        if self.bert_model.faiss_index is None:
            print("BERT document embeddings not created or FAISS index not built. Cannot perform semantic expansion.")
            return raw_query_text # Return original query if not set up

        # Retrieve top documents based on BERT similarity
        bert_top_docs = self.bert_model.search_faiss(query_embedding, k=top_n_docs)

        if not bert_top_docs:
            print("No relevant documents found for BERT semantic expansion.")
            return raw_query_text

        # Analyze terms in top documents. This is tricky with BERT embeddings directly.
        # A common approach is to find keywords in these documents, or average their embeddings
        # to create a 'concept vector', then find closest words in vocabulary to that vector.
        # This requires a mapping from BERT embedding space back to terms, which is complex.

        # Simpler approach: If we have access to the *original raw text* of the top documents,
        # we can extract keywords (e.g., using TF-IDF on these documents only, or textrank).
        # For conceptual: let's assume we can get the raw text of the top docs
        # and then extract some 'important' words.
        
        # This part is highly conceptual and typically involves
        # more advanced NLP techniques (e.g., keyphrase extraction,
        # or word embeddings for the vocabulary to find closest words to averaged document embeddings).

        # For demonstration: let's just use a placeholder for semantic expansion.
        # In a real system, you'd likely use a technique like:
        # 1. Averaging embeddings of top documents.
        # 2. Finding nearest neighbors in the word embedding space to this average.
        # This implies having word embeddings for your vocabulary.

        # As a simplified placeholder: let's say we get the document objects for top docs
        # and extract their most frequent terms after preprocessing.
        expanded_query_terms = list(self.bert_model.tokenizer.tokenize(raw_query_text)) # Initial tokens from BERT tokenizer
        
        # This needs access to the original `docs` list to get raw text by doc_id
        # For simplicity, let's assume `data_loader` can retrieve a doc by ID.
        # This would require `data_loader` to store the raw docs, or pass it around.
        
        # For now, this is a placeholder. Real semantic expansion is more involved.
        # You'd typically find semantic neighbors of the query embedding in the document space
        # and potentially identify terms that are highly associated with those document regions.
        print("Semantic query expansion is complex. Returning original query for now.")
        return raw_query_text # Currently returns original raw query as string

    def optimize_query(self, query, method="none", **kwargs):
        """
        Applies chosen query optimization method.
        Args:
            query (Query): The query object (from data_loader).
            method (str): "none", "spelling_correction", "spelling_correction_with_expansion", "tfidf_expansion", "bert_semantic_expansion", "cluster_restricted".
            **kwargs: Additional parameters for methods.
        """
        processed_query_vsm = self.inverted_index.preprocessor.preprocess_query(query.text)
        raw_query_text = query.text

        if method == "spelling_correction":
            confidence_threshold = kwargs.get('confidence_threshold', 0.6)
            return self.correct_spelling(processed_query_vsm, confidence_threshold)
        elif method == "spelling_correction_with_expansion":
            confidence_threshold = kwargs.get('confidence_threshold', 0.6)
            top_n_docs = kwargs.get('top_n_docs', 5)
            return self.spelling_correction_with_expansion(processed_query_vsm, top_n_docs, confidence_threshold)
        elif method == "tfidf_expansion":
            return self.query_expansion_tfidf(processed_query_vsm, **kwargs)
        elif method == "bert_semantic_expansion":
            query_embedding = self.bert_model.create_query_embedding(raw_query_text)
            return self.query_expansion_bert_semantic(query_embedding, raw_query_text, **kwargs)
        elif method == "cluster_restricted":
            if self.clusterer is None:
                raise ValueError("Clusterer not provided for cluster_restricted optimization.")
            query_embedding = self.bert_model.create_query_embedding(raw_query_text)
            nearest_cluster_id = self.clusterer.find_nearest_cluster(query_embedding)
            # Return a list of doc_ids that are relevant for this cluster
            # The retrieval system would then only search within these documents
            print(f"Restricting search to documents in cluster {nearest_cluster_id}")
            return self.clusterer.get_documents_in_cluster(nearest_cluster_id)
        elif method == "none":
            return processed_query_vsm # Return original processed query for VSM search
        else:
            raise ValueError(f"Unknown query optimization method: {method}")

