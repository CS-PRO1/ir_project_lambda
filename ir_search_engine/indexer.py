import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Union
from tqdm import tqdm # For progress bars

class InvertedIndex:
    def __init__(self):
        # self.index stores: {term: {doc_id: term_frequency_in_doc}}
        self.index: Dict[str, Dict[Union[int, str], int]] = defaultdict(dict)
        # self.documents stores a set of all unique document IDs that have been indexed
        self.documents: Set[Union[int, str]] = set()

    def build_index(self, documents: Dict[Union[int, str], List[str]]):
        """
        Builds the inverted index from a dictionary of documents.

        :param documents: A dictionary where keys are doc_ids (int or str)
                          and values are lists of preprocessed terms (tokens).
        """
        print("Building Inverted Index...")
        self.index.clear() # Clear existing index if any
        self.documents.clear() # Clear existing document IDs

        for doc_id, terms in tqdm(documents.items(), desc="Indexing documents"):
            self.documents.add(doc_id) # Add document ID to our set of all documents
            term_counts = defaultdict(int)
            for term in terms:
                term_counts[term] += 1
            
            for term, count in term_counts.items():
                self.index[term][doc_id] = count # Store term frequency for this document

    def get_postings(self, term: str) -> Dict[Union[int, str], int]:
        """
        Retrieves the postings list (documents and their term frequencies) for a given term.

        :param term: The term to look up.
        :return: A dictionary of {doc_id: term_frequency}. Returns empty dict if term not found.
        """
        return self.index.get(term, {})

    def get_all_terms(self) -> List[str]:
        """
        Returns a list of all unique terms in the index.
        """
        return list(self.index.keys())

    # --- ADDED THIS METHOD ---
    def get_all_documents(self) -> List[Union[int, str]]:
        """
        Returns a list of all unique document IDs present in the index.
        """
        return list(self.documents)
    # --- END ADDED METHOD ---

    def get_total_documents(self) -> int:
        """
        Returns the total number of unique documents in the index.
        """
        return len(self.documents)

    def save(self, filepath: str):
        """
        Saves the inverted index to a JSON file.

        :param filepath: The path to the JSON file where the index will be saved.
        """
        # Convert set of documents to list for JSON serialization
        json_serializable_documents = list(self.documents) 
        
        # Convert any integer keys in self.index[term] to strings for JSON,
        # and also handle defaultdicts for serialization.
        # This is important for JSON compatibility.
        serializable_index = {}
        for term, postings in self.index.items():
            serializable_index[term] = {str(doc_id): tf for doc_id, tf in postings.items()}

        data = {
            'index': serializable_index,
            'documents': json_serializable_documents # Save the list of documents
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        # print(f"Inverted index saved to {filepath}") # Already printed in main.py

    def load(self, filepath: str):
        """
        Loads the inverted index from a JSON file.

        :param filepath: The path to the JSON file to load the index from.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load documents and convert string keys back to original type (int or str)
        self.documents = set()
        for doc_id_str in data['documents']:
            # Attempt to convert to int if it's a digit string, otherwise keep as string
            self.documents.add(int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str)

        # Load index and convert string doc_ids back to original type (int or str)
        self.index = defaultdict(dict)
        for term, postings_str_keys in data['index'].items():
            for doc_id_str, tf in postings_str_keys.items():
                # Convert doc_id string back to its original type (int or str)
                original_doc_id = int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str
                self.index[term][original_doc_id] = tf
        # print(f"Inverted index loaded from {filepath}") # Already printed in main.py


# Example Usage (for direct testing of indexer.py)
if __name__ == "__main__":
    print("--- Testing InvertedIndex (Standalone) ---")

    # Sample documents with preprocessed terms (list of tokens)
    # Using mixed integer and string IDs for robust testing
    sample_docs = {
        1: ["quick", "brown", "fox", "jump"],
        2: ["lazy", "dog", "sleep"],
        "doc_abc": ["brown", "dog", "quick"],
        4: ["fox", "sleep", "dog"]
    }

    # Build the index
    idx = InvertedIndex()
    idx.build_index(sample_docs)

    # Test get_postings
    print("\nPostings for 'dog':", idx.get_postings('dog'))
    print("Postings for 'fox':", idx.get_postings('fox'))
    print("Postings for 'unknown_term':", idx.get_postings('unknown_term'))

    # Test get_all_terms
    print("\nAll terms:", idx.get_all_terms())

    # Test get_all_documents (NEW)
    all_docs = idx.get_all_documents()
    print("\nAll documents (IDs):", all_docs)
    print("Types of document IDs:", [type(d) for d in all_docs]) # Should show int and str

    # Test get_total_documents
    print("\nTotal documents:", idx.get_total_documents())

    # Test saving and loading
    test_filepath = "test_inverted_index.json"
    idx.save(test_filepath)
    print(f"\nIndex saved to {test_filepath}")

    loaded_idx = InvertedIndex()
    loaded_idx.load(test_filepath)
    print(f"Index loaded from {test_filepath}")

    print("\nLoaded index - Postings for 'dog':", loaded_idx.get_postings('dog'))
    loaded_all_docs = loaded_idx.get_all_documents()
    print("Loaded index - All documents (IDs):", loaded_all_docs)
    print("Loaded index - Types of document IDs:", [type(d) for d in loaded_all_docs])

    # Clean up
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
        print(f"Cleaned up {test_filepath}")