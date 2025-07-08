import numpy as np
from collections import defaultdict
import math
from typing import Dict, List, Tuple, Union
import time
from tqdm import tqdm

# Assuming these are in the same parent directory for relative imports
from preprocessing import TextPreprocessor
from indexer import InvertedIndex


class VectorSpaceModel:
    def __init__(self, inverted_index: InvertedIndex, preprocessor: TextPreprocessor):
        """
        Initializes the VectorSpaceModel.

        :param inverted_index: An instance of the InvertedIndex class.
        :param preprocessor: An instance of the TextPreprocessor class.
        """
        self.inverted_index = inverted_index
        self.preprocessor = preprocessor
        self.document_freq = self._calculate_document_frequencies()
        self.total_documents = len(self.inverted_index.get_all_documents())
        self.doc_lengths = self._calculate_document_lengths()
        print("Vector Space Model initialized.")


    def _calculate_document_frequencies(self) -> Dict[str, int]:
        """
        Calculates the document frequency (DF) for each term in the index.
        DF(t) = number of documents in which term t appears.

        :return: A dictionary of term: document_frequency.
        """
        df = {}
        for term, postings_list in self.inverted_index.index.items():
            df[term] = len(postings_list)
        return df

    def _calculate_document_lengths(self) -> Dict[Union[int, str], float]:
        """
        Calculates the Euclidean (L2) norm of the TF-IDF vector for each document.
        This is used for cosine similarity normalization.

        :return: A dictionary of doc_id: length.
        """
        doc_lengths = defaultdict(float)
        # Iterate through each term and its postings list
        # We need term frequencies within each document to calculate TF-IDF
        # The InvertedIndex stores postings as {term: {doc_id: tf}}
        
        # This part requires iterating over all documents and their terms, which can be slow.
        # We assume inverted_index.get_all_documents() returns original doc_ids
        
        # A more efficient way to calculate lengths might be during index building
        # or by iterating through the inverted index directly if it stores TF for each doc.
        # The current InvertedIndex stores postings as {term: {doc_id: tf}}
        
        # To compute lengths, we need IDF and TF per term per document.
        # IDF is based on global DF: idf(t) = log(N/DF(t))
        
        # Let's re-implement this by iterating through the index to get term freqs
        # and then calculating TF-IDF components.
        
        # Get all unique document IDs
        all_doc_ids = self.inverted_index.get_all_documents()
        doc_vectors = defaultdict(dict) # {doc_id: {term: tf_idf_score}}

        # Calculate TF-IDF for all terms in all documents
        for term, postings in tqdm(self.inverted_index.index.items(), desc="Calculating TF-IDF for document lengths"):
            if term not in self.document_freq: # Should not happen, but a safeguard
                continue
            idf = math.log(self.total_documents / self.document_freq[term])
            
            for doc_id, tf in postings.items():
                # Simple TF-IDF: tf * idf. Can use log-tf etc.
                tf_idf_score = tf * idf
                doc_vectors[doc_id][term] = tf_idf_score
        
        # Calculate Euclidean length for each document vector
        for doc_id, vector in tqdm(doc_vectors.items(), desc="Calculating document vector lengths"):
            length_sq = sum(score**2 for score in vector.values())
            doc_lengths[doc_id] = math.sqrt(length_sq)
            
        return doc_lengths


    def search(self, query_text: str, top_k: int = 10, add_ngrams: bool = False) -> List[Tuple[Union[int, str], float]]:
        """
        Performs a search using the Vector Space Model.

        :param query_text: The search query.
        :param top_k: The number of top documents to return.
        :param add_ngrams: Whether to add bigrams and trigrams to the query (for experimental purposes).
        :return: A list of (doc_id, score) tuples, sorted by score.
        """
        # Preprocess the query
        preprocessed_query_terms = self.preprocessor.preprocess_query(
            query_text, use_stemming=False, use_lemmatization=True, add_ngrams=add_ngrams
        )

        if not preprocessed_query_terms:
            print("No documents contain any query terms. Returning empty results.")
            return []

        # Calculate TF for query terms
        query_term_freq = defaultdict(int)
        for term in preprocessed_query_terms:
            query_term_freq[term] += 1

        # Calculate TF-IDF for query
        query_vector = {}
        for term, tf in query_term_freq.items():
            if term in self.document_freq:
                # Use log-frequency weighting for query TF
                # idf = log(N/DF)
                idf = math.log(self.total_documents / self.document_freq[term])
                query_vector[term] = (1 + math.log(tf)) * idf
            else:
                query_vector[term] = 0 # Term not in any document, so IDF is 0 or undefined

        # Calculate query vector length for normalization
        query_length = math.sqrt(sum(score**2 for score in query_vector.values()))
        if query_length == 0:
            return [] # No valid query terms

        # Identify candidate documents (documents containing any query term)
        candidate_doc_ids = set()
        for term in query_vector.keys():
            if term in self.inverted_index.index:
                for doc_id in self.inverted_index.index[term].keys():
                    candidate_doc_ids.add(doc_id)
        
        if not candidate_doc_ids:
            print("No documents contain any query terms. Returning empty results.")
            return []

        # Calculate cosine similarity for candidate documents
        similarities = {}
        
        # Using tqdm for progress bar during similarity calculation
        for doc_id in tqdm(candidate_doc_ids, desc="Calculating similarities"):
            if self.doc_lengths[doc_id] == 0: # Avoid division by zero for empty docs
                similarities[doc_id] = 0
                continue
            
            dot_product = 0
            for term, query_tfidf in query_vector.items():
                if term in self.inverted_index.index and doc_id in self.inverted_index.index[term]:
                    doc_tf = self.inverted_index.index[term][doc_id]
                    # Document term's TF-IDF: tf * idf (using same IDF as query)
                    doc_tfidf = doc_tf * math.log(self.total_documents / self.document_freq[term])
                    dot_product += query_tfidf * doc_tfidf
            
            similarity = dot_product / (query_length * self.doc_lengths[doc_id])
            similarities[doc_id] = similarity

        # Sort results by similarity score in descending order
        sorted_results = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_results[:top_k]:
            # --- START DEBUG PRINT (Added here) ---
            print(f"DEBUG_VSM: Original Doc ID from Inverted Index: {doc_id}, Type: {type(doc_id)}")
            # --- END DEBUG PRINT ---
            results.append((doc_id, score)) # Use the doc_id directly

        return results

# Example Usage (for direct testing of retrieval_model.py)
if __name__ == "__main__":
    print("--- Testing VectorSpaceModel (Standalone) ---")

    # 1. Setup a dummy TextPreprocessor
    preprocessor = TextPreprocessor(language='english')

    # 2. Setup dummy documents
    sample_docs_list = [
        ("doc1", "The quick brown fox jumps over the lazy dog. The dog is very lazy."),
        ("doc2", "A brown dog is always good. Fox and dog are friends."),
        ("doc3", "Quick brown foxes are cunning, but dogs are loyal. Quick fox."),
        (101, "This is a document with an integer ID. Integer IDs are common."), # Integer ID example
        (102, "Another document with an integer ID. Just for testing types."),
    ]
    
    # Preprocess documents for index building
    # Note: In a real scenario, InvertedIndex usually takes preprocessed texts.
    # For this test, we'll preprocess them here directly.
    
    # Prepare documents dict for InvertedIndex as {doc_id: raw_text}
    raw_docs_dict = {doc_id: text for doc_id, text in sample_docs_list}

    # Preprocess texts for indexer (lemmatization)
    preprocessed_texts = preprocessor.preprocess_documents(
        [text for doc_id, text in sample_docs_list],
        use_stemming=False, use_lemmatization=True, add_ngrams=False,
        desc="Preprocessing for VSM init"
    )
    # Map preprocessed texts back to their original doc IDs
    preprocessed_docs_for_index = {
        sample_docs_list[i][0]: preprocessed_texts[i] 
        for i in range(len(sample_docs_list))
    }

    # 3. Build a dummy InvertedIndex
    inverted_index = InvertedIndex()
    inverted_index.build_index(preprocessed_docs_for_index)

    # 4. Initialize VectorSpaceModel
    vsm = VectorSpaceModel(inverted_index, preprocessor)

    # 5. Perform a search
    query = "brown dog"
    print(f"\nSearching for: '{query}'")
    
    # Example for integer ID doc_id mapping check
    if 101 in vsm.inverted_index.get_all_documents():
        print(f"Doc 101 found in index. Its type: {type(101)}") # Should be int
    
    start_time = time.time()
    results = vsm.search(query, top_k=3)
    end_time = time.time()

    print(f"\nResults for '{query}' (found in {end_time - start_time:.4f} seconds):")
    for doc_id, score in results:
        # For this standalone test, we don't have a direct 'documents_text' source
        # so we'll just print the ID and score.
        print(f"  Doc ID: {doc_id}, Score: {score:.4f}")

    # Test with a query that might not have direct matches
    query_no_match = "gato" # Spanish for cat
    print(f"\nSearching for: '{query_no_match}'")
    results_no_match = vsm.search(query_no_match, top_k=3)
    if not results_no_match:
        print("  No results found, as expected.")