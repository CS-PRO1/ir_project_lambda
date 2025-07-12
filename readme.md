# Information Retrieval Search Engine

A comprehensive Information Retrieval (IR) search engine system with multiple retrieval models, document clustering, query optimization, and thorough evaluation capabilities. The system provides both a FastAPI backend server and a Streamlit web interface for easy interaction.

## 🏗️ System Architecture

### Modular Design

The system is organized into clean, modular components for better maintainability and scalability:

#### 📁 **Core Application** (`core/`)
- **`server.py`** - FastAPI backend server with all API endpoints
- **`client_app.py`** - Streamlit frontend application  
- **`main.py`** - Command-line interface for the search engine

#### 📁 **Data Processing** (`data_processing/`)
- **`data_loader.py`** - Downloads and manages datasets from `ir_datasets`
- **`preprocessing.py`** - Text preprocessing, tokenization, spell checking

#### 📁 **Retrieval Models** (`retrieval_models/`)
- **`indexer.py`** - Inverted index construction for fast retrieval
- **`retrieval_model.py`** - Vector Space Model (TF-IDF) implementation
- **`bert_retrieval.py`** - BERT-based semantic search model
- **`hybrid_retrieval.py`** - Hybrid ranking combining TF-IDF and BERT

#### 📁 **Query Processing** (`query_processing/`)
- **`query_optimizer.py`** - Pseudo-Relevance Feedback (PRF) for query expansion

#### 📁 **Evaluation** (`evaluation/`)
- **`evaluator.py`** - Comprehensive IR metrics (MAP, Recall, Precision@10, MRR)

#### 📁 **Clustering** (`clustering/`)
- **`clusterer.py`** - K-means clustering with PCA dimensionality reduction

### Core Components

1. **Data Loading & Management**
   - Supports `antique_train` and `beir_webist_touche2020` datasets
   - Handles documents, queries, and relevance judgments
   - Automatic dataset caching for faster subsequent runs

2. **Text Preprocessing**
   - Text cleaning and normalization
   - Tokenization and stop word removal
   - Stemming and lemmatization
   - N-gram generation
   - Spell correction using `pyspellchecker`

3. **Retrieval Models**
   - **Vector Space Model (VSM)**: Traditional TF-IDF based retrieval
   - **BERT Model**: Semantic search using BERT embeddings
   - **Hybrid Model**: Combines VSM and BERT for enhanced performance

4. **Advanced Features**
   - **Document Clustering**: Groups similar documents using K-means
   - **Query Optimization**: Pseudo-Relevance Feedback (PRF) for query expansion
   - **Evaluation**: Comprehensive IR metrics (MAP, Recall, Precision@10, MRR)

5. **Web Interface**
   - **FastAPI Backend**: RESTful API for all operations
   - **Streamlit Frontend**: User-friendly web interface

### How It Works

1. **Data Loading**: The system loads documents, queries, and relevance judgments from IR datasets
2. **Preprocessing**: Text is cleaned, tokenized, and normalized for indexing
3. **Indexing**: An inverted index is built for fast document retrieval
4. **Model Training**: BERT embeddings are generated and clustering is performed
5. **Search**: Users can search using different models (VSM, BERT, Hybrid)
6. **Evaluation**: Comprehensive metrics are calculated to assess performance

## 🚀 Installation Guide

### Prerequisites

- Python 3.8 or higher
- Git
- At least 4GB RAM (8GB recommended for BERT models)

### Step-by-Step Installation

1. **Clone the Repository**
    ```bash
   git clone <repository-url>
   cd ir_project_lambda
    ```

2. **Create Virtual Environment**
    ```bash
   # On Windows
    python -m venv venv
   .venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Required Data**
   ```bash
   # The system will automatically download datasets on first run
   # This may take several minutes depending on your internet connection
   ```

## 🏃‍♂️ Running the System

### Option 1: PowerShell Scripts (Windows - Recommended)

For Windows users, we provide convenient PowerShell scripts:

1. **Start Server Only**
   ```powershell
   .\start_server.ps1
   ```

2. **Start UI Only** (requires server to be running)
   ```powershell
   .\start_ui.ps1
   ```

3. **Start Both Components** (opens separate windows)
   ```powershell
   .\start_all.ps1
   ```

### Option 2: Manual Commands

1. **Start the Backend Server**
   ```bash
   # Make sure your virtual environment is activated
   uvicorn ir_search_engine.core.server:app --host 127.0.0.1 --port 8000 --reload
   ```
   The server will:
   - Load and preprocess datasets
   - Build inverted indexes
   - Generate BERT embeddings
   - Perform document clustering
   - Initialize all models

2. **Start the Streamlit Frontend**
   ```bash
   # In a new terminal (with virtual environment activated)
   streamlit run ir_search_engine.core.client_app
   ```

3. **Access the Web Interface**
   - Open your browser and go to `http://localhost:8501`
   - Select your dataset and start searching!

### Option 2: Command Line Interface

1. **Run the Main Script**
   ```bash
   python -m ir_search_engine.core.main
   ```

2. **Follow the Interactive Prompts**
   - Choose your dataset
   - Select search model
   - Enter queries
   - View results and metrics

## 📊 Features

### Search Capabilities

- **Multiple Models**: TF-IDF, BERT, and Hybrid retrieval
- **Spell Correction**: Automatic query spelling correction
- **Clustered Search**: Search within document clusters for better relevance
- **Query Optimization**: PRF-based query expansion

### Evaluation Metrics

The system calculates four key IR metrics:

1. **MAP (Mean Average Precision)**: Measures overall retrieval effectiveness
2. **Recall**: Proportion of relevant documents retrieved
3. **Precision@10**: Precision of top 10 results
4. **MRR (Mean Reciprocal Rank)**: How early the first relevant document appears

### Web Interface Features

- **Interactive Search**: Real-time search with multiple models
- **Results Display**: Clean table format with document previews
- **Evaluation Dashboard**: Comprehensive metrics table with performance summaries
- **Export Functionality**: Download results as CSV files
- **Document Viewing**: Full-text document display

## 🔧 Configuration

### Server Settings

- **Default URL**: `http://127.0.0.1:8000`
- **Default Port**: 8000
- **Dataset Storage**: `./data/` directory

### Model Parameters

- **BERT Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Clustering**: 5 clusters by default
- **PRF Parameters**: Configurable via web interface

## 📁 Project Structure

```
ir_project_lambda/
├── data/                          # Dataset storage
│   ├── antique_train/             # Antique dataset
│   ├── beir_webis-touche2020/    # WebIS dataset
│   └── indexes/                   # Cached indexes and embeddings
├── ir_search_engine/
│   ├── core/                      # Application core
│   │   ├── server.py              # FastAPI backend
│   │   ├── client_app.py          # Streamlit frontend
│   │   ├── main.py                # CLI entry point
│   │   └── __init__.py
│   ├── data_processing/           # Data loading and preprocessing
│   │   ├── data_loader.py         # Dataset management
│   │   ├── preprocessing.py       # Text preprocessing
│   │   └── __init__.py
│   ├── retrieval_models/          # Search and retrieval models
│   │   ├── indexer.py             # Inverted index
│   │   ├── retrieval_model.py     # VSM implementation
│   │   ├── bert_retrieval.py      # BERT model
│   │   ├── hybrid_retrieval.py    # Hybrid model
│   │   └── __init__.py
│   ├── query_processing/          # Query optimization
│   │   ├── query_optimizer.py     # PRF implementation
│   │   └── __init__.py
│   ├── evaluation/                # Performance evaluation
│   │   ├── evaluator.py           # Evaluation metrics
│   │   └── __init__.py
│   ├── clustering/                # Document clustering
│   │   ├── clusterer.py           # K-means clustering
│   │   └── __init__.py
│   └── __init__.py               # Main package exports
├── requirements.txt               # Python dependencies
├── start_server.ps1              # PowerShell script to start server
├── start_ui.ps1                  # PowerShell script to start UI
├── start_all.ps1                 # PowerShell script to start both
└── README.md                     # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **Server Connection Error**
   - Ensure the backend server is running on port 8000
   - Check if the URL in the web interface matches your server

2. **Memory Issues**
   - BERT models require significant RAM
   - Consider using smaller models or reducing batch sizes

3. **Dataset Loading Errors**
   - Check internet connection for dataset downloads
   - Ensure sufficient disk space in the `data/` directory

4. **CUDA/GPU Issues**
   - The system automatically falls back to CPU if CUDA is unavailable
   - BERT models will work on CPU (slower but functional)

### Performance Tips

- **First Run**: Initial setup may take 15-30 minutes for dataset processing
- **Subsequent Runs**: Much faster due to cached indexes and embeddings
- **Memory Usage**: Close other applications if experiencing memory issues
- **Network**: Ensure stable internet for initial dataset downloads

## 📚 Module Usage Examples

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

## 🙏 Acknowledgments

- Built with FastAPI, Streamlit, and Transformers
- Uses datasets from `ir_datasets`
- Modular architecture for easy maintenance and extension
