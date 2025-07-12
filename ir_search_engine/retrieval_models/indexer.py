import os
import json
from collections import defaultdict
from typing import Dict, List, Set, Union
from tqdm import tqdm

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[Union[int, str], int]] = defaultdict(dict)
        self.documents: Set[Union[int, str]] = set()
        self.doc_lengths: Dict[Union[int, str], int] = {} 
        self.doc_to_terms: Dict[Union[int, str], Set[str]] = defaultdict(set)

    def build_index(self, documents: Dict[Union[int, str], List[str]]):
        print("Building Inverted Index...")
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()
        self.doc_to_terms.clear()

        for doc_id, terms in tqdm(documents.items(), desc="Indexing documents"):
            self.documents.add(doc_id)
            self.doc_lengths[doc_id] = len(terms)
            self.doc_to_terms[doc_id] = set(terms)
            
            term_counts = defaultdict(int)
            for term in terms:
                term_counts[term] += 1
            
            for term, count in term_counts.items():
                self.index[term][doc_id] = count

    def get_postings(self, term: str) -> Dict[Union[int, str], int]:
        return self.index.get(term, {})

    def get_all_terms(self) -> List[str]:
        return list(self.index.keys())

    def get_all_documents(self) -> List[Union[int, str]]:
        return list(self.documents)

    def get_total_documents(self) -> int:
        return len(self.documents)

    def get_document_terms(self, doc_id: Union[int, str]) -> Set[str]:
        return self.doc_to_terms.get(doc_id, set())

    def save(self, filepath: str):
        json_serializable_documents = list(self.documents) 
        
        serializable_index = {}
        for term, postings in self.index.items():
            serializable_index[term] = {str(doc_id): tf for doc_id, tf in postings.items()}

        serializable_doc_lengths = {str(doc_id): length for doc_id, length in self.doc_lengths.items()}
        serializable_doc_to_terms = {str(doc_id): list(terms_set) for doc_id, terms_set in self.doc_to_terms.items()}

        data = {
            'index': serializable_index,
            'documents': json_serializable_documents,
            'doc_lengths': serializable_doc_lengths,
            'doc_to_terms': serializable_doc_to_terms
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = set()
        for doc_id_str in data['documents']:
            self.documents.add(int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str)

        self.index = defaultdict(dict)
        for term, postings_str_keys in data['index'].items():
            for doc_id_str, tf in postings_str_keys.items():
                original_doc_id = int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str
                self.index[term][original_doc_id] = tf
        
        self.doc_lengths = {}
        if 'doc_lengths' in data:
            for doc_id_str, length in data['doc_lengths'].items():
                original_doc_id = int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str
                self.doc_lengths[original_doc_id] = length
        else:
            print("Warning: 'doc_lengths' not found in loaded index file. TF-IDF calculations might be affected.")

        self.doc_to_terms = defaultdict(set)
        if 'doc_to_terms' in data:
            for doc_id_str, terms_list in data['doc_to_terms'].items():
                original_doc_id = int(doc_id_str) if str(doc_id_str).isdigit() else doc_id_str
                self.doc_to_terms[original_doc_id] = set(terms_list)
        else:
            print("Warning: 'doc_to_terms' not found in loaded index file. Document term lookups might be affected.")