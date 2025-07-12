import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import time

class DocumentClusterer:
    def __init__(self, n_clusters=5, random_state=42, use_pca=False, pca_components=50):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.document_clusters = {}
        self.cluster_centroids = None
        self.use_pca = use_pca
        self.pca = None
        self.pca_components = pca_components

    def cluster_documents(self, doc_embeddings, doc_ids):
        print(f"Clustering documents into {self.n_clusters} clusters...")

        data_to_cluster = doc_embeddings
        if self.use_pca and doc_embeddings.shape[1] > self.pca_components:
            print(f"Applying PCA for dimensionality reduction to {self.pca_components} components.")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            data_to_cluster = self.pca.fit_transform(doc_embeddings)
            print(f"PCA explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.2f}")

        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(data_to_cluster)
        self.cluster_centroids = self.kmeans_model.cluster_centers_

        self.document_clusters = {doc_id: label for doc_id, label in zip(doc_ids, cluster_labels)}
        print("Document clustering complete.")

        return self.document_clusters

    def get_cluster_for_document(self, doc_id):
        return self.document_clusters.get(doc_id)

    def get_documents_in_cluster(self, cluster_id):
        return [doc_id for doc_id, label in self.document_clusters.items() if label == cluster_id]

    def find_nearest_cluster(self, query_embedding):
        if self.kmeans_model is None:
            raise ValueError("Clustering model not trained. Call cluster_documents first.")
        
        query_data = query_embedding
        if self.pca:
            query_data = self.pca.transform(query_embedding)
        
        distances = np.linalg.norm(self.cluster_centroids - query_data, axis=1)
        nearest_cluster_id = np.argmin(distances)
        return nearest_cluster_id

    def save_clusterer(self, filepath):
        print(f"Saving clusterer to {filepath}...")
        start_time_save = time.time()
        try:
            with open(filepath, 'wb') as f:
                pickle.dump((self.kmeans_model, self.document_clusters, self.cluster_centroids, self.pca), f)
            print(f"Clusterer saved in {time.time() - start_time_save:.2f} seconds.")
            return True
        except Exception as e:
            print(f"Error saving clusterer: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def load_clusterer(self, filepath):
        if not os.path.exists(filepath):
            print(f"Clusterer file not found at {filepath}")
            return False
        try:
            with open(filepath, 'rb') as f:
                self.kmeans_model, self.document_clusters, self.cluster_centroids, self.pca = pickle.load(f)
            print(f"Clusterer loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading clusterer: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
