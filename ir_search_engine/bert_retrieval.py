import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import os
import json
from typing import Dict, Union
from tqdm import tqdm # For progress bars

class BERTRetrievalModel:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the BERTRetrievalModel.
        Loads a pre-trained Sentence-BERT model and sets up for embedding generation.

        :param model_name: The name of the pre-trained Sentence-BERT model to use.
        (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        """
        print(f"Loading BERT model: {model_name}...")
        
        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the SentenceTransformer model, explicitly moving it to the determined device
        # This is the crucial part to ensure it uses the GPU from the start if 'cuda' is available.
        self.model = SentenceTransformer(model_name, device=self.device)
        
        print(f"Model loaded and moved to {self.device}.")
        
        # Store document embeddings and mappings
        self.document_embeddings_matrix: np.ndarray = None # Stores all document embeddings
        self.doc_id_map: Dict[Union[int, str], int] = {} # Maps original doc_id to internal index
        self.reverse_doc_id_map: Dict[int, Union[int, str]] = {} # Maps internal index to original doc_id
        self.documents_text: Dict[Union[int, str], str] = {} # Stores original raw text by doc_id

    def index_documents(self, documents: Dict[Union[int, str], str]):
        """
        Generates embeddings for a collection of documents and indexes them.

        :param documents: A dictionary of documents {doc_id: raw_text}.
        """
        print(f"Generating embeddings for {len(documents)} documents...")

        self.documents_text = documents # Store original texts
        
        # Sort documents by their IDs to ensure consistent indexing order
        sorted_doc_ids = sorted(documents.keys())
        sorted_doc_texts = [documents[doc_id] for doc_id in sorted_doc_ids]

        # Generate embeddings in batches.
        # batch_size can be tuned based on GPU memory. 32 is a common default.
        # show_progress_bar=True will display a tqdm progress bar.
        embeddings = self.model.encode(sorted_doc_texts, 
                                       convert_to_tensor=True, 
                                       show_progress_bar=True,
                                       batch_size=32)
        
        # Move embeddings to CPU and convert to NumPy array for storage
        self.document_embeddings_matrix = embeddings.cpu().numpy()

        # Create mapping from original doc_id to their index in the embeddings matrix
        self.doc_id_map = {doc_id: i for i, doc_id in enumerate(sorted_doc_ids)}
        self.reverse_doc_id_map = {i: doc_id for i, doc_id in enumerate(sorted_doc_ids)}

        print(f"Indexed {len(self.doc_id_map)} documents.")

    def search(self, query_text: str, top_k: int = 10) -> list:
        """
        Performs a semantic search using BERT embeddings.

        :param query_text: The query string.
        :param top_k: The number of top results to return.
        :return: A list of (doc_id, score) tuples, sorted by score descending.
        """
        if self.document_embeddings_matrix is None or len(self.doc_id_map) == 0:
            print("Error: Documents not indexed. Call index_documents() first.")
            return []
        
        # Generate embedding for the query
        # Ensure the query embedding is also on the correct device
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).to(self.device)

        # Calculate cosine similarity between query embedding and all document embeddings
        # Ensure document embeddings are also on the correct device for calculation
        # self.document_embeddings_matrix is a numpy array, convert to tensor and move to device
        doc_embeddings_tensor = torch.from_numpy(self.document_embeddings_matrix).to(self.device)
        
        # cos_sim function can handle tensor inputs directly
        similarities = cos_sim(query_embedding, doc_embeddings_tensor)[0] # Get the first row (for single query)
        
        # Move similarities back to CPU and convert to NumPy for sorting
        similarities = similarities.cpu().numpy()

        # Get top_k results
        # Use argsort to get indices of top similarities, then reverse for descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            doc_id = self.reverse_doc_id_map[i]
            score = similarities[i]
            results.append((doc_id, float(score))) # Convert score to float for consistency

        return results

    def save_embeddings(self, base_filepath: str):
        """
        Saves the document embeddings and their mapping to disk.

        :param base_filepath: The base path (without extension) for saving.
                              e.g., 'my_bert_embeddings' will save to 'my_bert_embeddings.npy'
                              and 'my_bert_embeddings_map.json'.
        """
        if self.document_embeddings_matrix is None:
            print("No embeddings to save.")
            return

        npy_path = f"{base_filepath}.npy"
        json_path = f"{base_filepath}_map.json"
        text_path = f"{base_filepath}_text.json" # To save original document texts

        np.save(npy_path, self.document_embeddings_matrix)
        
        # Save mappings and original texts
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert non-string keys to string for JSON serialization if necessary
            # (e.g., if doc_ids are integers, json_map will store them as strings)
            json_serializable_map = {str(k): v for k, v in self.doc_id_map.items()}
            json.dump({'doc_id_map': json_serializable_map, 'reverse_doc_id_map': self.reverse_doc_id_map}, f, indent=4)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            json_serializable_texts = {str(k): v for k, v in self.documents_text.items()}
            json.dump(json_serializable_texts, f, indent=4)

        print(f"BERT embeddings saved to {npy_path}, {json_path} and {text_path}")

    def load_embeddings(self, base_filepath: str):
        """
        Loads document embeddings and their mapping from disk.

        :param base_filepath: The base path (without extension) from which to load.
        """
        npy_path = f"{base_filepath}.npy"
        json_path = f"{base_filepath}_map.json"
        text_path = f"{base_filepath}_text.json"

        if not os.path.exists(npy_path) or not os.path.exists(json_path) or not os.path.exists(text_path):
            print(f"Could not find all required BERT embedding files: {npy_path}, {json_path}, {text_path}")
            self.document_embeddings_matrix = None
            self.doc_id_map = {}
            self.reverse_doc_id_map = {}
            self.documents_text = {}
            return

        self.document_embeddings_matrix = np.load(npy_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert doc_id_map keys back to original type if they were integers
            # Assumes original IDs are integers if they parse as such
            self.doc_id_map = {int(k) if k.isdigit() else k: v for k, v in data['doc_id_map'].items()}
            self.reverse_doc_id_map = {v: (int(k) if k.isdigit() else k) for k, v in data['doc_id_map'].items()} # Reconstruct reverse map carefully

        with open(text_path, 'r', encoding='utf-8') as f:
            loaded_texts = json.load(f)
            self.documents_text = {int(k) if k.isdigit() else k: v for k, v in loaded_texts.items()} # Convert keys back

        print(f"BERT embeddings loaded from {npy_path} and {json_path}. {len(self.doc_id_map)} documents.")

# Example Usage (for direct testing of bert_retrieval.py)
if __name__ == "__main__":
    print("--- Testing BERTRetrievalModel (Standalone) ---")

    # Sample documents
    sample_docs = {
        0: "The quick brown fox jumps over the lazy dog.",
        1: "Never jump over a lazy dog, it might bark.",
        2: "Foxes are cunning, dogs are loyal. The dog is brown."
    }

    # Initialize model
    bert_retriever = BERTRetrievalModel()

    # Index documents
    bert_retriever.index_documents(sample_docs)

    # Save embeddings
    base_file = "my_bert_embeddings"
    bert_retriever.save_embeddings(base_file)

    # --- Test Search (before loading to verify current state) ---
    print("\nSearching for: 'a sleepy puppy'")
    results = bert_retriever.search('a sleepy puppy', top_k=3)
    print("Top results:")
    for doc_id, score in results:
        # Access original text directly from the model's stored documents_text
        print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Original Text: '{bert_retriever.documents_text.get(doc_id, 'N/A')}'")

    # --- Test Loading Embeddings ---
    print(f"\nLoading embeddings from {base_file}.npy...")
    loaded_bert_retriever = BERTRetrievalModel() # Create a new instance
    loaded_bert_retriever.load_embeddings(base_file)

    # Verification after loading
    print("\nVerification after loading, searching for 'a sleepy puppy':")
    loaded_results = loaded_bert_retriever.search('a sleepy puppy', top_k=3)
    print("Top results:")
    for doc_id, score in loaded_results:
        print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Original Text: '{loaded_bert_retriever.documents_text.get(doc_id, 'N/A')}'")

    # Clean up created files
    print("\nCleaning up temporary files...")
    if os.path.exists(f"{base_file}.npy"):
        os.remove(f"{base_file}.npy")
    if os.path.exists(f"{base_file}_map.json"):
        os.remove(f"{base_file}_map.json")
    if os.path.exists(f"{base_file}_text.json"):
        os.remove(f"{base_file}_text.json")
    print("Cleanup complete.")