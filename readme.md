# Comprehensive IR Search Engine

This project implements a comprehensive Information Retrieval (IR) search engine with various advanced features, including diverse document representations, optimized indexing, document clustering, and query optimization. It leverages datasets from `ir_datasets` and provides thorough evaluation capabilities.

## Project Structure

The project is organized into the following modules:

-   `main.py`: The main entry point for running the search engine, orchestrating the different stages.
-   `data_loader.py`: Handles downloading and loading datasets from `ir_datasets`.
-   `preprocessing.py`: Contains functions for text cleaning, tokenization, stop word removal, stemming, lemmatization, and N-gram generation.
-   `indexing.py`: Responsible for building and managing the inverted index.
-   `representation.py`: Implements different document and query representation models (VSM TF-IDF, BERT Embeddings, Hybrid).
-   `clustering.py`: Provides functionality for document clustering.
-   `query_optimization.py`: Implements techniques for optimizing user queries.
-   `evaluation.py`: Contains functions to calculate standard IR metrics (MAP, Recall, Precision@10, MRR).
-   `utils.py`: A utility module for common helper functions (e.g., file I/O, multi-threading helpers).

## Features

-   **Dataset Handling**: Utilizes `ir_datasets` for `antique_train` and `beir_webist_touche2020`.
-   **Text Preprocessing**: Robust cleaning, stop word removal, stemming, lemmatization, and support for 2-word and 3-word terms.
-   **Optimized Indexing**: Efficient inverted index implementation for fast retrieval.
-   **Document Representations**:
    -   Vector Space Model (VSM) with TF-IDF weighting.
    -   BERT Embedding-based representation for semantic understanding.
    -   Hybrid representation combining VSM and BERT.
-   **Document Clustering**: Groups similar documents for enhanced Browse or query refinement.
-   **Query Optimization**: Techniques to improve query effectiveness.
-   **Comprehensive Evaluation**: Calculates MAP, Recall, Precision@10, and MRR, with and without clustering/query optimization.
-   **Performance Optimization**: Utilizes multi-threading for various stages to maximize efficiency.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/ir-search-engine.git](https://github.com/yourusername/ir-search-engine.git)
    cd ir-search-engine
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    (You will need to create `requirements.txt` with `ir_datasets`, `nltk`, `scikit-learn`, `transformers`, `torch` or `tensorflow`, `faiss-cpu` (for efficient similarity search with embeddings), `numpy`, `scipy`, `tqdm`.)

## Usage

A typical workflow would involve:

1.  **Load Data**:
    ```python
    from data_loader import DataLoader
    data_loader = DataLoader()
    corpus_antique, queries_antique, qrels_antique = data_loader.load_antique_train()
    corpus_webist, queries_webist, qrels_webist = data_loader.load_beir_webist_touche2020()
    ```
2.  **Preprocess Data**:
    ```python
    from preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    processed_corpus = preprocessor.preprocess_documents(corpus_antique)
    ```
3.  **Build Index**:
    ```python
    from indexing import InvertedIndex
    index = InvertedIndex()
    index.build_index(processed_corpus)
    ```
4.  **Represent Documents/Queries**:
    ```python
    from representation import VSMRepresentation, BERTEmbeddingRepresentation, HybridRepresentation
    vsm_model = VSMRepresentation(index.get_vocabulary())
    doc_vectors_vsm = vsm_model.create_document_vectors(processed_corpus)

    bert_model = BERTEmbeddingRepresentation()
    doc_embeddings_bert = bert_model.create_document_embeddings(corpus_antique) # Use raw text for BERT
    ```
5.  **Run Search and Evaluate**:
    ```python
    from evaluation import IREvaluator
    from query_optimization import QueryOptimizer

    # VSM Evaluation
    retrieved_docs_vsm = # ... perform search using VSM ...
    evaluator = IREvaluator()
    map_vsm, recall_vsm, p10_vsm, mrr_vsm = evaluator.evaluate(qrels_antique, retrieved_docs_vsm)

    # With Query Optimization and Clustering (Conceptual)
    # query_optimizer = QueryOptimizer(...)
    # optimized_queries = query_optimizer.optimize_queries(queries_antique)
    # clusterer = DocumentClusterer(...)
    # clusters = clusterer.cluster_documents(doc_embeddings_bert)
    # ... then run search and evaluate again ...
    ```

For detailed usage and to run the complete pipeline, refer to `main.py` and individual module files.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.