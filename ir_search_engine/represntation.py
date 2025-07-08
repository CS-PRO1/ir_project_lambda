import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import faiss # For efficient similarity search with embeddings
import concurrent.futures
from tqdm import tqdm

class VSMRepresentation:
    def __init__(self, vocabulary=None):
        self.vectorizer = TfidfVectorizer(vocabulary=vocabulary)
        self.tfidf_matrix = None
        self.doc_ids = []

    def create_document_vectors(self, processed_corpus):
        """
        Creates TF-IDF vectors for documents.
        `processed_corpus` should be a dict: {doc_id: [token1, token2, ...]}
        """
        print("Creating VSM TF-IDF document vectors...")
        texts = [" ".join(tokens) for tokens in processed_corpus.values()]
        self.doc_ids = list(processed_corpus.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix

    def create_query_vector(self, processed_query):
        """Creates a TF-IDF vector for a query."""
        return self.vectorizer.transform([" ".join(processed_query)])

    def get_document_vectors(self):
        return self.tfidf_matrix, self.doc_ids

    def get_vocabulary(self):
        return self.vectorizer.vocabulary_

class BERTEmbeddingRepresentation:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.doc_embeddings = None
        self.doc_ids = []
        self.faiss_index = None # For efficient similarity search

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def create_embeddings(self, texts, batch_size=32):
        """Creates BERT embeddings for a list of texts."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT embeddings"):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings.append(sentence_embeddings.cpu().numpy())
        return np.vstack(embeddings)

    def create_document_embeddings(self, documents):
        """
        Creates BERT embeddings for documents.
        `documents` should be a list of Document namedtuples (raw text).
        """
        print("Creating BERT document embeddings...")
        self.doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.text for doc in documents]
        self.doc_embeddings = self.create_embeddings(texts)

        # Build FAISS index for efficient cosine similarity search
        dimension = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity
        faiss.normalize_L2(self.doc_embeddings) # Normalize for cosine similarity
        self.faiss_index.add(self.doc_embeddings)

        print(f"BERT embeddings shape: {self.doc_embeddings.shape}")
        return self.doc_embeddings

    def create_query_embedding(self, query_text):
        """Creates a BERT embedding for a query."""
        embedding = self.create_embeddings([query_text])
        faiss.normalize_L2(embedding) # Normalize for cosine similarity
        return embedding

    def search_faiss(self, query_embedding, k=100):
        """Searches the FAISS index for top-k similar documents."""
        D, I = self.faiss_index.search(query_embedding, k)
        # D contains distances/scores, I contains indices in self.doc_embeddings
        results = []
        for i, score in zip(I[0], D[0]):
            results.append((self.doc_ids[i], score))
        return results # (doc_id, similarity_score)


class HybridRepresentation:
    def __init__(self, vsm_model, bert_model):
        self.vsm_model = vsm_model
        self.bert_model = bert_model

    def retrieve_hybrid(self, processed_query_vsm, raw_query_bert, k=100, alpha=0.5):
        """
        Combines results from VSM and BERT using a parallel approach and Reciprocal Rank Fusion (RRF).
        alpha controls the weight of each component (not directly used in RRF, but conceptually for blending).
        """
        print("Performing hybrid retrieval...")

        # VSM Retrieval
        query_vector_vsm = self.vsm_model.create_query_vector(processed_query_vsm)
        tfidf_matrix, doc_ids_vsm = self.vsm_model.get_document_vectors()

        # Calculate cosine similarity for VSM
        # Note: Scipy's cosine_similarity expects sparse matrices
        from sklearn.metrics.pairwise import cosine_similarity
        vsm_scores = cosine_similarity(query_vector_vsm, tfidf_matrix).flatten()

        vsm_results = []
        for i, score in enumerate(vsm_scores):
            vsm_results.append((doc_ids_vsm[i], score))
        vsm_results.sort(key=lambda x: x[1], reverse=True)
        vsm_results_ranked = vsm_results[:k]

        # BERT Retrieval
        query_embedding_bert = self.bert_model.create_query_embedding(raw_query_bert)
        bert_results_faiss = self.bert_model.search_faiss(query_embedding_bert, k=k)

        # Merge using Reciprocal Rank Fusion (RRF)
        # RRF formula: score = sum(1 / (k + rank)) for each document across all rankings
        fused_scores = defaultdict(float)
        rank_constant = 60 # A common constant for RRF

        # Add VSM ranks
        for rank, (doc_id, _) in enumerate(vsm_results_ranked):
            fused_scores[doc_id] += 1.0 / (rank_constant + rank + 1) # +1 because rank is 0-indexed

        # Add BERT ranks
        for rank, (doc_id, _) in enumerate(bert_results_faiss):
            fused_scores[doc_id] += 1.0 / (rank_constant + rank + 1)

        final_ranked_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        return final_ranked_docs # Returns (doc_id, fused_score) tuples
