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
        self.tokenizer = None
        self.model = None
        try:
            self.model = SentenceTransformer(model_name)
            self.tokenizer = self.model.tokenizer
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"Model loaded and moved to {self.device}.")
            print(f"Model loaded successfully: {self.model is not None}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.model = None
            self.device = torch.device('cpu')
            print("Falling back to CPU for BERT model.")
            print(f"Model loading failed: {self.model is None}")

        self.document_embeddings_matrix: Optional[np.ndarray] = None
        self.doc_id_map: Dict[Union[int, str], int] = {}
        self.reverse_doc_id_map: Dict[int, Union[int, str]] = {}
        self.documents_text: Dict[Union[int, str], str] = {}

    def encode_documents(self, documents: Dict[Union[int, str], str], batch_size: int = 32) -> np.ndarray:
        if not self.model:
            print("BERT model not loaded. Cannot encode documents.")
            return np.array([])
        
        print(f"Starting document encoding...")
        print(f"  - Number of documents: {len(documents)}")
        print(f"  - Model available: {self.model is not None}")
        
        doc_texts = list(documents.values())
        doc_ids = list(documents.keys())

        embeddings = []
        try:
            for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents with BERT"):
                batch_texts = doc_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                embeddings.append(batch_embeddings)
            
            all_embeddings = np.concatenate(embeddings, axis=0)
            print(f"Successfully encoded {all_embeddings.shape[0]} documents with {all_embeddings.shape[1]} dimensions")

            self.doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}
            self.reverse_doc_id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            self.documents_text = documents
            
            return all_embeddings
        except Exception as e:
            print(f"Error during document encoding: {e}")
            return np.array([])

    def encode_query(self, query: str) -> np.ndarray:
        if not self.model:
            print("BERT model not loaded. Cannot encode query.")
            return np.array([])
        try:
            encoded = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
            print(f"Query encoded successfully. Shape: {encoded.shape}")
            return encoded
        except Exception as e:
            print(f"Error encoding query: {e}")
            return np.array([])

    def index_documents(self, documents: Dict[Union[int, str], str]):
        print(f"Generating BERT embeddings for {len(documents)} documents...")
        print(f"Model status before encoding: {self.model is not None}")
        self.document_embeddings_matrix = self.encode_documents(documents)
        if self.document_embeddings_matrix is not None and self.document_embeddings_matrix.size > 0:
            print(f"Generated {self.document_embeddings_matrix.shape[0]} embeddings of dimension {self.document_embeddings_matrix.shape[1]}.")
        else:
            print("Warning: No embeddings were generated!")

    def save_embeddings(self, filepath_base: str):
        if self.document_embeddings_matrix is None or not self.doc_id_map:
            print("No embeddings to save.")
            return

        npy_path = f"{filepath_base}.npy"
        map_json_path = f"{filepath_base}_map.json"
        text_json_path = f"{filepath_base}_text.json"

        np.save(npy_path, self.document_embeddings_matrix)
        with open(map_json_path, 'w') as f:
            serializable_doc_id_map = {str(k): v for k, v in self.doc_id_map.items()}
            json.dump(serializable_doc_id_map, f)
        with open(text_json_path, 'w') as f:
            serializable_documents_text = {str(k): v for k, v in self.documents_text.items()}
            json.dump(serializable_documents_text, f)
        print(f"BERT embeddings saved to {npy_path}, {map_json_path}, and {text_json_path}.")

    def load_embeddings(self, filepath_base: str) -> bool:
        npy_path = f"{filepath_base}.npy"
        map_json_path = f"{filepath_base}_map.json"
        text_json_path = f"{filepath_base}_text.json"

        if not (os.path.exists(npy_path) and os.path.exists(map_json_path) and os.path.exists(text_json_path)):
            print(f"One or more BERT embedding files not found for {filepath_base}. Skipping load.")
            return False
        
        try:
            self.document_embeddings_matrix = np.load(npy_path)
            with open(map_json_path, 'r') as f:
                loaded_map = json.load(f)
                self.doc_id_map = {int(k) if k.isdigit() else k: v for k, v in loaded_map.items()}
            self.reverse_doc_id_map = {v: k for k, v in self.doc_id_map.items()}
            with open(text_json_path, 'r') as f:
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
        print(f"BERT search debug:")
        print(f"  - Model exists: {self.model is not None}")
        print(f"  - Embeddings matrix exists: {self.document_embeddings_matrix is not None}")
        if self.document_embeddings_matrix is not None:
            print(f"  - Embeddings shape: {self.document_embeddings_matrix.shape}")
        print(f"  - Doc ID map size: {len(self.doc_id_map)}")
        
        if not self.model or self.document_embeddings_matrix is None:
            print("BERT model or document embeddings not loaded.")
            return []

        if encode_query_again:
            query_embedding = self.encode_query(query)
        else:
            query_embedding = self.encode_query(query)

        if query_embedding.size == 0:
            return []

        if candidate_doc_ids:
            filtered_indices = [
                self.doc_id_map[doc_id] for doc_id in candidate_doc_ids
                if doc_id in self.doc_id_map
            ]
            if not filtered_indices:
                return []
            
            search_embeddings = self.document_embeddings_matrix[filtered_indices]
            search_reverse_doc_id_map = {i: self.reverse_doc_id_map[original_idx] 
                                          for i, original_idx in enumerate(filtered_indices)}
        else:
            search_embeddings = self.document_embeddings_matrix
            search_reverse_doc_id_map = self.reverse_doc_id_map
        
        if search_embeddings.shape[0] == 0:
            return []

        cosine_scores = util.cos_sim(query_embedding, search_embeddings)[0].cpu().numpy()

        top_k_indices = np.argpartition(cosine_scores, -min(top_k, len(cosine_scores)))[-min(top_k, len(cosine_scores)):]
        
        scores_for_sorting = cosine_scores[top_k_indices]
        sorted_top_k_indices = top_k_indices[np.argsort(-scores_for_sorting)]

        results = []
        for idx in sorted_top_k_indices:
            original_doc_id = search_reverse_doc_id_map[idx]
            score = cosine_scores[idx]
            results.append((original_doc_id, float(score)))

        return results

    def get_document_text_by_internal_id(self, internal_id: int) -> Optional[str]:
        original_doc_id = self.reverse_doc_id_map.get(internal_id)
        if original_doc_id is not None:
            return self.documents_text.get(original_doc_id)
        return None