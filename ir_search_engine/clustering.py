import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

class DocumentClusterer:
    def __init__(self, n_clusters=5, random_state=42, use_pca=False, pca_components=50):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.document_clusters = {} # {doc_id: cluster_id}
        self.cluster_centroids = None
        self.use_pca = use_pca
        self.pca = None
        self.pca_components = pca_components

    def cluster_documents(self, doc_embeddings, doc_ids):
        """
        Clusters documents based on their embeddings.
        Args:
            doc_embeddings (np.array): NxM numpy array of document embeddings.
            doc_ids (list): List of document IDs corresponding to the embeddings.
        """
        print(f"Clustering documents into {self.n_clusters} clusters...")

        data_to_cluster = doc_embeddings
        if self.use_pca and doc_embeddings.shape[1] > self.pca_components:
            print(f"Applying PCA for dimensionality reduction to {self.pca_components} components.")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            data_to_cluster = self.pca.fit_transform(doc_embeddings)
            print(f"PCA explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.2f}")

        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10) # n_init for robustness
        cluster_labels = self.kmeans_model.fit_predict(data_to_cluster)
        self.cluster_centroids = self.kmeans_model.cluster_centers_

        self.document_clusters = {doc_id: label for doc_id, label in zip(doc_ids, cluster_labels)}
        print("Document clustering complete.")

        # Evaluate clustering (optional, for insights)
        try:
            silhouette_avg = silhouette_score(data_to_cluster, cluster_labels)
            print(f"Silhouette Score: {silhouette_avg:.3f} (higher is better, typically between -1 and 1)")
        except Exception as e:
            print(f"Could not calculate Silhouette Score (requires >= 2 clusters and > n_samples): {e}")

        return self.document_clusters

    def get_cluster_for_document(self, doc_id):
        return self.document_clusters.get(doc_id)

    def get_documents_in_cluster(self, cluster_id):
        return [doc_id for doc_id, label in self.document_clusters.items() if label == cluster_id]

    def find_nearest_cluster(self, query_embedding):
        """Finds the closest cluster centroid to a given query embedding."""
        if self.kmeans_model is None:
            raise ValueError("Clustering model not trained. Call cluster_documents first.")
        
        query_data = query_embedding
        if self.pca: # Apply same PCA transformation if used during clustering
            query_data = self.pca.transform(query_embedding)
        
        distances = np.linalg.norm(self.cluster_centroids - query_data, axis=1)
        nearest_cluster_id = np.argmin(distances)
        return nearest_cluster_id

    def save_clusterer(self, filepath):
        """Saves the clusterer model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.kmeans_model, self.document_clusters, self.cluster_centroids, self.pca), f)
        print(f"Clusterer saved to {filepath}")

    def load_clusterer(self, filepath):
        """Loads the clusterer model from a file."""
        if not os.path.exists(filepath):
            print(f"Clusterer file not found at {filepath}")
            return False
        with open(filepath, 'rb') as f:
            self.kmeans_model, self.document_clusters, self.cluster_centroids, self.pca = pickle.load(f)
        print(f"Clusterer loaded from {filepath}")
        return True

