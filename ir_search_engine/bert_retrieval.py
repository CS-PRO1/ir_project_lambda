import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import os
import json

class BERTRetrievalModel:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading BERT model: {model_name}...")
        self.tokenizer = None # Initialized by SentenceTransformer
        self.model = None # Initialized by SentenceTransformer
        try:
            self.model = SentenceTransformer(model_name)
            self.tokenizer = self.model.tokenizer # Access tokenizer from the loaded model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"Model loaded and moved to {self.device}.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.model = None
            self.device = torch.device('cpu') # Fallback to CPU if CUDA fails or not available
            print("Falling back to CPU for BERT model.")

        self.document_embeddings_matrix: Optional[np.ndarray] = None
        self.doc_id_map: Dict[Union[int, str], int] = {} # Maps original doc_id to internal numpy index
        self.reverse_doc_id_map: Dict[int, Union[int, str]] = {} # Maps internal numpy index back to original doc_id
        self.documents_text: Dict[Union[int, str], str] = {} # Stores raw document texts

    def encode_documents(self, documents: Dict[Union[int, str], str], batch_size: int = 32) -> np.ndarray:
        if not self.model:
            print("BERT model not loaded. Cannot encode documents.")
            return np.array([])
        
        doc_texts = list(documents.values())
        doc_ids = list(documents.keys())

        embeddings = []
        for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents with BERT"):
            batch_texts = doc_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)

        # Update mappings
        self.doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        self.reverse_doc_id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.documents_text = documents # Store the raw texts
        
        return all_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        if not self.model:
            print("BERT model not loaded. Cannot encode query.")
            return np.array([])
        return self.model.encode(query, convert_to_numpy=True).reshape(1, -1) # Ensure 2D array

    def index_documents(self, documents: Dict[Union[int, str], str]):
        """Generates and stores embeddings for a collection of documents."""
        print(f"Generating BERT embeddings for {len(documents)} documents...")
        self.document_embeddings_matrix = self.encode_documents(documents)
        print(f"Generated {self.document_embeddings_matrix.shape[0]} embeddings of dimension {self.document_embeddings_matrix.shape[1]}.")

    def save_embeddings(self, filepath_base: str):
        if self.document_embeddings_matrix is None or not self.doc_id_map:
            print("No embeddings to save.")
            return

        npy_path = f"{filepath_base}.npy"
        map_json_path = f"{filepath_base}_map.json"
        text_json_path = f"{filepath_base}_text.json" # New: Save raw texts

        np.save(npy_path, self.document_embeddings_matrix)
        with open(map_json_path, 'w') as f:
            # Convert non-string keys to string if necessary before saving JSON
            # This handles both int and str doc_ids
            serializable_doc_id_map = {str(k): v for k, v in self.doc_id_map.items()}
            json.dump(serializable_doc_id_map, f)
        with open(text_json_path, 'w') as f: # Save raw texts
            serializable_documents_text = {str(k): v for k, v in self.documents_text.items()}
            json.dump(serializable_documents_text, f)
        print(f"BERT embeddings saved to {npy_path}, {map_json_path}, and {text_json_path}.")

    def load_embeddings(self, filepath_base: str) -> bool:
        npy_path = f"{filepath_base}.npy"
        map_json_path = f"{filepath_base}_map.json"
        text_json_path = f"{filepath_base}_text.json" # New: Load raw texts

        if not (os.path.exists(npy_path) and os.path.exists(map_json_path) and os.path.exists(text_json_path)):
            print(f"One or more BERT embedding files not found for {filepath_base}. Skipping load.")
            return False
        
        try:
            self.document_embeddings_matrix = np.load(npy_path)
            with open(map_json_path, 'r') as f:
                loaded_map = json.load(f)
                # Convert string keys back to original type if they were integers
                self.doc_id_map = {int(k) if k.isdigit() else k: v for k, v in loaded_map.items()}
            self.reverse_doc_id_map = {v: k for k, v in self.doc_id_map.items()}
            with open(text_json_path, 'r') as f: # Load raw texts
                loaded_texts = json.load(f)
                self.documents_text = {int(k) if k.isdigit() else k: v for k, v in loaded_texts.items()}

            print(f"BERT embeddings loaded from {npy_path}, {map_json_path} and {text_json_path}. {len(self.doc_id_map)} documents.")
            return True
        except Exception as e:
            print(f"Error loading BERT embeddings: {e}")
            self.document_embeddings_matrix = None
            self.doc_id_map = {}
            self.reverse_doc_id_map = {}
            self.documents_text = {}
            return False

    def search(self, query: str, top_k: int = 10, encode_query_again: bool = True,
               candidate_doc_ids: Optional[List[Union[int, str]]] = None) -> List[tuple[Union[int, str], float]]:
        """
        Performs a BERT-based search.
        Args:
            query (str): The search query.
            top_k (int): The number of top documents to retrieve.
            encode_query_again (bool): If True, encodes the query. Set to False if query is already encoded.
            candidate_doc_ids (Optional[List[Union[int, str]]]): If provided, search only within these document IDs.
        Returns:
            List[Tuple[Union[int, str], float]]: A list of (doc_id, score) tuples, sorted by score.
        """
        if not self.model or self.document_embeddings_matrix is None:
            print("BERT model or document embeddings not loaded.")
            return []

        if encode_query_again:
            query_embedding = self.encode_query(query)
        else:
            # Assume 'query' argument is already the encoded query embedding if encode_query_again is False
            # This is a bit ambiguous with 'query: str' type hint, but often used for internal calls.
            # For this purpose, it's safer to ensure query is encoded before passing if this is an external call.
            # If called internally with an actual embedding, one might cast or assert type.
            # For simplicity, we'll stick to 'query: str' and re-encode, or rely on external caller to handle.
            # For evaluator, the query will always be a string and will be encoded.
            query_embedding = self.encode_query(query) # Still encode to be safe for `evaluator.py` usage

        if query_embedding.size == 0:
            return []

        # Determine which embeddings to search through
        if candidate_doc_ids:
            # Filter embeddings and corresponding doc_ids
            filtered_indices = [
                self.doc_id_map[doc_id] for doc_id in candidate_doc_ids
                if doc_id in self.doc_id_map # Ensure the candidate doc_id actually exists
            ]
            if not filtered_indices:
                return []
            
            search_embeddings = self.document_embeddings_matrix[filtered_indices]
            # Create a temporary mapping for results to original doc_ids
            search_reverse_doc_id_map = {i: self.reverse_doc_id_map[original_idx] 
                                          for i, original_idx in enumerate(filtered_indices)}
        else:
            # Search through all documents
            search_embeddings = self.document_embeddings_matrix
            search_reverse_doc_id_map = self.reverse_doc_id_map
        
        if search_embeddings.shape[0] == 0:
            return []

        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, search_embeddings)[0].cpu().numpy()

        # Get top_k results
        # Use argpartition for partial sort, then full sort on the top_k
        top_k_indices = np.argpartition(cosine_scores, -min(top_k, len(cosine_scores)))[-min(top_k, len(cosine_scores)):]
        
        # Sort the top_k indices by score in descending order
        # Need to re-index scores based on top_k_indices for sorting
        scores_for_sorting = cosine_scores[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-scores_for_sorting)]

        results = []
        for idx in sorted_top_k_indices:
            original_doc_id = search_reverse_doc_id_map[idx]
            score = cosine_scores[idx]
            results.append((original_doc_id, float(score)))

        return results

    # The get_document_text_by_internal_id method remains the same if you have it
    # This is not directly used by `search` itself, but potentially by external components.
    def get_document_text_by_internal_id(self, internal_id: int) -> Optional[str]:
        original_doc_id = self.reverse_doc_id_map.get(internal_id)
        if original_doc_id is not None:
            return self.documents_text.get(original_doc_id)
        return None