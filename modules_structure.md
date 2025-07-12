# IR Search Engine - Modular Structure

## Overview
The IR Search Engine has been restructured into a clean, modular architecture organized by functionality. Each module is self-contained and has clear responsibilities.

## Module Structure

### 📁 `core/` - Application Core
Contains the main application files and entry points.
- **`server.py`** - FastAPI backend server with all API endpoints
- **`client_app.py`** - Streamlit frontend application
- **`main.py`** - Command-line interface for the search engine
- **`__init__.py`** - Exports main application components

### 📁 `data_processing/` - Data Loading and Preprocessing
Handles all data loading, parsing, and text preprocessing operations.
- **`data_loader.py`** - Loads datasets (antique/train, beir/webis-touche2020)
- **`preprocessing.py`** - Text preprocessing, tokenization, spell checking
- **`__init__.py`** - Exports data loading and preprocessing classes

### 📁 `retrieval_models/` - Search and Retrieval Models
Contains all retrieval models and indexing functionality.
- **`indexer.py`** - Inverted index implementation for TF-IDF
- **`retrieval_model.py`** - Vector Space Model (TF-IDF) implementation
- **`bert_retrieval.py`** - BERT-based semantic search model
- **`hybrid_retrieval.py`** - Hybrid ranking combining TF-IDF and BERT
- **`__init__.py`** - Exports all retrieval model classes

### 📁 `query_processing/` - Query Optimization
Handles query expansion and optimization techniques.
- **`query_optimizer.py`** - Pseudo-relevance feedback (PRF) implementation
- **`__init__.py`** - Exports query optimization functionality

### 📁 `evaluation/` - Performance Evaluation
Contains evaluation metrics and model comparison functionality.
- **`evaluator.py`** - Evaluation metrics (MAP, Recall, Precision@10, MRR)
- **`__init__.py`** - Exports evaluation functions

### 📁 `clustering/` - Document Clustering
Handles document clustering for improved search performance.
- **`clusterer.py`** - K-means clustering with PCA dimensionality reduction
- **`__init__.py`** - Exports clustering functionality

## Key Benefits of This Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easy to locate and modify specific functionality
3. **Reusability**: Modules can be imported and used independently
4. **Scalability**: Easy to add new models or functionality to appropriate modules
5. **Testing**: Each module can be tested in isolation

## Import Examples

```python
# Import main application components
from ir_search_engine import app, main

# Import data processing components
from ir_search_engine import DataLoader, TextPreprocessor

# Import retrieval models
from ir_search_engine import VectorSpaceModel, BERTRetrievalModel, HybridRanker

# Import evaluation and clustering
from ir_search_engine import evaluate_models, DocumentClusterer
```

## Module Dependencies

- **`core/`** depends on all other modules
- **`retrieval_models/`** depends on `data_processing/`
- **`query_processing/`** depends on `data_processing/` and `retrieval_models/`
- **`evaluation/`** depends on `data_processing/`, `retrieval_models/`, `query_processing/`, and `clustering/`
- **`clustering/`** is independent but used by other modules

This structure provides a clean, organized codebase that is easy to understand, maintain, and extend. 