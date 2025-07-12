import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, List, Union, Optional
from tqdm import tqdm
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ir_search_engine.data_loader import DataLoader, Document, Query, Qrel
from ir_search_engine.preprocessing import TextPreprocessor
from ir_search_engine.indexer import InvertedIndex
from ir_search_engine.retrieval_model import VectorSpaceModel
from ir_search_engine.bert_retrieval import BERTRetrievalModel
from ir_search_engine.hybrid_retrieval import HybridRanker
from ir_search_engine.query_optimizer import QueryOptimizer
from ir_search_engine.evaluator import evaluate_models 
from ir_search_engine.clusterer import DocumentClusterer

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
INDEXES_DIR = os.path.join(DATA_DIR, 'indexes')
os.makedirs(INDEXES_DIR, exist_ok=True)

class DatasetComponents:
    def __init__(self, name: str, docs: List[Document], queries: List[Query], qrels: List[Qrel]):
        self.name = name
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.raw_documents_dict: Dict[Union[int, str], str] = {}
        self.inverted_index: Optional[InvertedIndex] = None
        self.vsm_model: Optional[VectorSpaceModel] = None
        self.bert_model: Optional[BERTRetrievalModel] = None
        self.hybrid_model: Optional[HybridRanker] = None
        self.document_clusterer: Optional[DocumentClusterer] = None 

global_datasets_cache: Dict[str, DatasetComponents] = {}
global_preprocessor: Optional[TextPreprocessor] = None
global_query_optimizer: Optional[QueryOptimizer] = None

app = FastAPI(
    title="IR Search Engine API",
    description="Backend for the Information Retrieval Search Engine, handling data loading, indexing, and search models. Evaluation metrics include MAP, Recall, Precision at 10, and MRR.",
    version="1.0.0",
)

def get_raw_documents_dict(docs_list: List[Document]) -> Dict[Union[int, str], str]:
    return {doc.doc_id: doc.text for doc in tqdm(docs_list, desc="Preparing raw document dict")}

def load_or_build_inverted_index(dataset_name: str, docs: List[Document], preprocessor: TextPreprocessor) -> InvertedIndex:
    index_filepath = os.path.join(INDEXES_DIR, f"{dataset_name}_inverted_index.json")
    index = InvertedIndex()

    if os.path.exists(index_filepath):
        print(f"Loading Inverted Index for {dataset_name}...")
        try:
            index.load(index_filepath)
            if index.index and index.doc_lengths:
                print(f"Inverted index loaded from {index_filepath}.")
                return index
            else:
                print(f"Loaded index was incomplete. Rebuilding...")
                if os.path.exists(index_filepath):
                    os.remove(index_filepath)
        except Exception as e:
            print(f"Error loading inverted index: {e}. Rebuilding...")
            if os.path.exists(index_filepath):
                os.remove(index_filepath)
    else:
        print(f"Inverted Index file not found. Building...")
    
    doc_texts_list = [doc.text for doc in docs]
    processed_docs_list = preprocessor.preprocess_documents(
        doc_texts_list, use_stemming=False, use_lemmatization=True, add_ngrams=False,
        desc="Preprocessing documents for indexing"
    )
    preprocessed_docs_dict = {
        docs[i].doc_id: processed_docs_list[i] 
        for i in tqdm(range(len(docs)), desc="Mapping preprocessed docs for indexing")
    }
    print("Building Inverted Index...")
    index.build_index(preprocessed_docs_dict)
    index.save(index_filepath)
    print(f"Inverted index built and saved to {index_filepath}")
    return index

def load_or_index_bert_embeddings(dataset_name: str, raw_documents_dict: Dict[Union[int, str], str]) -> BERTRetrievalModel:
    embeddings_filepath_base = os.path.join(INDEXES_DIR, f"{dataset_name}_bert_embeddings")
    
    bert_model = BERTRetrievalModel()

    npy_path = f"{embeddings_filepath_base}.npy"
    map_json_path = f"{embeddings_filepath_base}_map.json"
    text_json_path = f"{embeddings_filepath_base}_text.json"

    if os.path.exists(npy_path) and os.path.exists(map_json_path) and os.path.exists(text_json_path):
        print(f"Loading BERT embeddings for {dataset_name}...")
        try:
            bert_model.load_embeddings(embeddings_filepath_base)
            
            if (bert_model.document_embeddings_matrix is not None and 
                bert_model.document_embeddings_matrix.size > 0 and 
                bert_model.doc_id_map and 
                bert_model.reverse_doc_id_map and 
                bert_model.documents_text):
                
                print("BERT embeddings loaded successfully.")
                print(f"BERT model ready with {len(bert_model.doc_id_map)} embeddings.")
                return bert_model
            else:
                print(f"Loaded BERT embeddings were incomplete. Rebuilding...")
                for f in [npy_path, map_json_path, text_json_path]:
                    if os.path.exists(f):
                        os.remove(f)
        except Exception as e:
            print(f"Error loading BERT embeddings: {e}. Rebuilding...")
            for f in [npy_path, map_json_path, text_json_path]:
                if os.path.exists(f):
                    os.remove(f)
    else:
        print(f"BERT embeddings not found for {dataset_name}. Generating...")
    
    bert_model.index_documents(raw_documents_dict)
    bert_model.save_embeddings(embeddings_filepath_base)
    print(f"BERT embeddings generated and saved. BERT model ready with {len(bert_model.doc_id_map)} embeddings.")
    
    bert_model.documents_text = raw_documents_dict 
    
    return bert_model

@app.on_event("startup")
async def startup_event():
    print("--- Starting IR Search Engine Server ---")
    
    global global_preprocessor, global_query_optimizer, global_datasets_cache
    
    global_preprocessor = TextPreprocessor(language='english')
    global_query_optimizer = QueryOptimizer(global_preprocessor)

    data_loader = DataLoader(base_data_path=DATA_DIR)
    
    available_datasets_to_load = ['antique_train', 'beir_webist_touche2020']

    for dataset_name in available_datasets_to_load:
        print(f"\n--- Loading and initializing for dataset: {dataset_name} ---")
        docs, queries, qrels = None, None, None
        
        if dataset_name == 'antique_train':
            docs, queries, qrels = data_loader.load_antique_train()
        elif dataset_name == 'beir_webist_touche2020':
            docs, queries, qrels = data_loader.load_beir_webist_touche2020()
        
        if docs is None:
            print(f"Failed to load documents for {dataset_name}. Skipping this dataset.")
            continue

        dataset_components = DatasetComponents(dataset_name, docs, queries, qrels)
        dataset_components.raw_documents_dict = get_raw_documents_dict(docs)

        dataset_components.inverted_index = load_or_build_inverted_index(
            dataset_name, docs, global_preprocessor
        )
        dataset_components.vsm_model = VectorSpaceModel(
            dataset_components.inverted_index, global_preprocessor
        )
        print(f"VSM model for {dataset_name} ready with {dataset_components.vsm_model.total_documents} documents.")

        if global_preprocessor and dataset_components.inverted_index and not global_preprocessor.spell.word_frequency.words():
            print(f"Loading SpellChecker vocabulary from {dataset_name} Inverted Index terms...")
            global_preprocessor.spell.word_frequency.load_words(dataset_components.inverted_index.get_all_terms())
            print(f"SpellChecker vocabulary loaded with {len(dataset_components.inverted_index.get_all_terms())} terms.")
        elif global_preprocessor and global_preprocessor.spell.word_frequency.words():
            print("SpellChecker vocabulary already loaded.")
        else:
            print("Warning: SpellChecker vocabulary not loaded from Inverted Index (Preprocessor or Index not ready).")

        dataset_components.bert_model = load_or_index_bert_embeddings(
            dataset_name, dataset_components.raw_documents_dict
        )
        print(f"BERT model for {dataset_name} ready with {len(dataset_components.bert_model.doc_id_map)} embeddings.")

        N_CLUSTERS = 5
        USE_PCA_FOR_CLUSTERING = True
        PCA_COMPONENTS_FOR_CLUSTERING = 10
    
        clusterer_filepath = os.path.join(INDEXES_DIR, f"{dataset_name}_clusterer.pkl")
        
        dataset_components.document_clusterer = DocumentClusterer(
            n_clusters=N_CLUSTERS, 
            use_pca=USE_PCA_FOR_CLUSTERING, 
            pca_components=PCA_COMPONENTS_FOR_CLUSTERING
        )
    
        if not dataset_components.document_clusterer.load_clusterer(clusterer_filepath):
            print(f"Clustering model not found for {dataset_name}. Performing clustering...")
            if dataset_components.bert_model and dataset_components.bert_model.document_embeddings_matrix is not None:
                doc_embeddings = dataset_components.bert_model.document_embeddings_matrix
                doc_ids = [dataset_components.bert_model.reverse_doc_id_map[i] 
                            for i in range(len(dataset_components.bert_model.doc_id_map))]
                
                start_time_cluster = time.time()
                dataset_components.document_clusterer.cluster_documents(
                    doc_embeddings, 
                    doc_ids
                )
                if dataset_components.document_clusterer.save_clusterer(clusterer_filepath):
                    print(f"Clustering for {dataset_name} completed and saved in {time.time() - start_time_cluster:.2f} seconds.")
                else:
                    print(f"Clustering for {dataset_name} completed but failed to save.")
            else:
                print(f"Warning: BERT embeddings not available for {dataset_name}. Skipping clustering.")
        else:
            print(f"Document clusterer for {dataset_name} loaded.")

        dataset_components.hybrid_model = HybridRanker(
            dataset_components.inverted_index, global_preprocessor, dataset_components.bert_model
        )
        print(f"Hybrid Ranker for {dataset_name} ready.")

        global_datasets_cache[dataset_name] = dataset_components
        print(f"Dataset '{dataset_name}' fully loaded and models initialized.")

    print("\n--- All datasets and models initialized. Server is ready to accept connections. ---")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    apply_spell_correction: bool = False

class SearchResult(BaseModel):
    doc_id: Union[int, str]
    score: float
    text_preview: str

class OptimizationRequest(BaseModel):
    query: str
    initial_search_model: str
    top_n_docs_for_prf: int = 5
    num_expansion_terms: int = 3

class FullDocumentResponse(BaseModel):
    doc_id: Union[int, str]
    text: str

class ClusterInfo(BaseModel):
    cluster_id: int
    num_documents: int

class DocumentClusterAssignment(BaseModel):
    doc_id: Union[int, str]
    cluster_id: int

class ClusterSearchRequest(BaseModel):
    query: str
    target_cluster_id: Optional[int] = None
    top_k: int = 10
    apply_spell_correction: bool = False

class EvaluationRequest(BaseModel):
    use_clustering_for_bert_evaluation: bool = False
    use_prf_for_evaluation: bool = False
    prf_initial_model: Optional[str] = None
    prf_top_n_docs: int = 5
    prf_num_expansion_terms: int = 3
    prf_final_model: Optional[str] = None

class EvaluationMetrics(BaseModel):
    model_name: str
    map: float
    recall: float
    precision_at_10: float
    mrr: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the IR Search Engine API!"}

@app.get("/datasets", response_model=List[str])
async def get_available_datasets():
    return list(global_datasets_cache.keys())

@app.post("/search/{dataset_name}/{model_type}", response_model=List[SearchResult])
async def search(dataset_name: str, model_type: str, request: SearchRequest):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")

    original_query = request.query
    processed_query = original_query

    print(f"\n--- New Search Request ---")
    print(f"Original Query: '{original_query}'")
    print(f"Model: {model_type}, Top K: {request.top_k}")
    print(f"Apply Spell Correction: {request.apply_spell_correction}")

    if request.apply_spell_correction and global_preprocessor:
        start_time_sc = time.time()
        corrected_query = global_preprocessor.correct_query_spelling(original_query)
        if corrected_query.lower() != original_query.lower():
            print(f"Spell Correction Applied: '{original_query}' -> '{corrected_query}'")
            processed_query = corrected_query
        else:
            print(f"Spell Correction: No changes to '{original_query}'")
        print(f"Spell correction took {time.time() - start_time_sc:.4f} seconds.")
    elif request.apply_spell_correction and not global_preprocessor:
        print("Warning: Spell correction requested but TextPreprocessor not initialized. Skipping.")

    results = []
    start_time_search = time.time()
    
    if model_type.lower() == 'tf-idf':
        if not dataset_info.vsm_model:
            raise HTTPException(status_code=500, detail="VSM model not initialized for this dataset.")
        results = dataset_info.vsm_model.search(processed_query, top_k=request.top_k)
    elif model_type.lower() == 'bert':
        if not dataset_info.bert_model:
            raise HTTPException(status_code=500, detail="BERT model not initialized for this dataset.")
        results = dataset_info.bert_model.search(processed_query, top_k=request.top_k)
    elif model_type.lower() == 'hybrid':
        if not dataset_info.hybrid_model:
            raise HTTPException(status_code=500, detail="Hybrid model not initialized for this dataset.")
        results = dataset_info.hybrid_model.hybrid_search(
            processed_query,
            top_k=request.top_k, 
            vsm_weight=0.1, 
            bert_weight=0.9, 
            top_k_bert_initial=200
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'tf-idf', 'bert', or 'hybrid'.")

    end_time_search = time.time()
    print(f"Final search on {dataset_name} using {model_type} for query '{processed_query}' took {end_time_search - start_time_search:.4f} seconds.")

    response_results = []
    for doc_id, score in results:
        original_text = dataset_info.raw_documents_dict.get(doc_id)
        if original_text is None:
            if isinstance(doc_id, str) and doc_id.isdigit():
                original_text = dataset_info.raw_documents_dict.get(int(doc_id))
            elif isinstance(doc_id, int):
                original_text = dataset_info.raw_documents_dict.get(str(doc_id))

        text_preview = (original_text[:200] + '...') if original_text else "Text not available."
        response_results.append(SearchResult(doc_id=doc_id, score=score, text_preview=text_preview))
    
    return response_results

@app.post("/optimize_query/{dataset_name}", response_model=str)
async def optimize_query(dataset_name: str, request: OptimizationRequest):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    
    if not global_query_optimizer:
        raise HTTPException(status_code=500, detail="Query Optimizer not initialized.")

    retrieval_model_for_prf = None
    if request.initial_search_model.lower() == 'tf-idf':
        retrieval_model_for_prf = dataset_info.vsm_model
    elif request.initial_search_model.lower() == 'bert':
        retrieval_model_for_prf = dataset_info.bert_model
    else:
        raise HTTPException(status_code=400, detail="Invalid initial_search_model for PRF. Choose 'TF-IDF' or 'BERT'.")

    if not retrieval_model_for_prf:
        raise HTTPException(status_code=500, detail=f"{request.initial_search_model} model not initialized for PRF.")

    expanded_query = global_query_optimizer.expand_query_with_prf(
        original_query=request.query,
        retrieval_model=retrieval_model_for_prf,
        raw_documents_dict=dataset_info.raw_documents_dict,
        top_n_docs_for_prf=request.top_n_docs_for_prf,
        num_expansion_terms=request.num_expansion_terms
    )
    return expanded_query

@app.get("/document/{dataset_name}/{doc_id}", response_model=FullDocumentResponse)
async def get_document_text(dataset_name: str, doc_id: Union[int, str]):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    
    text = dataset_info.raw_documents_dict.get(doc_id)
    if text is None:
        if isinstance(doc_id, str) and doc_id.isdigit():
            text = dataset_info.raw_documents_dict.get(int(doc_id))
        elif isinstance(doc_id, int):
            text = dataset_info.raw_documents_dict.get(str(doc_id))

    if text is None:
        raise HTTPException(status_code=404, detail=f"Document ID '{doc_id}' not found in dataset '{dataset_name}'.")
    
    return FullDocumentResponse(doc_id=doc_id, text=text)

@app.post("/check_spelling/")
async def check_spelling_endpoint(text: str):
    if not global_preprocessor:
        raise HTTPException(status_code=500, detail="TextPreprocessor not initialized.")
    
    original_tokens = global_preprocessor.preprocess_document(text)
    corrected_tokens = []
    corrections = {}
    
    for token in original_tokens:
        corrected_token = global_preprocessor.spell.correction(token)
        if corrected_token and corrected_token.lower() != token.lower():
            corrected_tokens.append(corrected_token)
            corrections[token] = corrected_token
        else:
            corrected_tokens.append(token)
            
    return {
        "original_text": text,
        "corrected_text": " ".join(corrected_tokens),
        "corrections_made": corrections
    }

@app.get("/cluster/{dataset_name}/info", response_model=Dict[int, ClusterInfo])
async def get_cluster_info(dataset_name: str):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    if not dataset_info.document_clusterer or not dataset_info.document_clusterer.kmeans_model:
        raise HTTPException(status_code=500, detail="Document clusterer not initialized for this dataset.")

    cluster_counts = defaultdict(int)
    for doc_id, cluster_id in dataset_info.document_clusterer.document_clusters.items():
        cluster_counts[cluster_id] += 1
    
    response_info = {}
    for cluster_id, count in sorted(cluster_counts.items()):
        response_info[cluster_id] = ClusterInfo(cluster_id=cluster_id, num_documents=count)
    
    return response_info

@app.get("/cluster/{dataset_name}/assignments", response_model=List[DocumentClusterAssignment])
async def get_document_cluster_assignments(dataset_name: str):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    if not dataset_info.document_clusterer or not dataset_info.document_clusterer.kmeans_model:
        raise HTTPException(status_code=500, detail="Document clusterer not initialized for this dataset.")

    assignments = [
        DocumentClusterAssignment(doc_id=doc_id, cluster_id=cluster_id)
        for doc_id, cluster_id in dataset_info.document_clusterer.document_clusters.items()
    ]
    return assignments

@app.post("/cluster/{dataset_name}/search", response_model=List[SearchResult])
async def search_with_clustering(dataset_name: str, request: ClusterSearchRequest):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    
    if not dataset_info.bert_model or not dataset_info.document_clusterer:
        raise HTTPException(status_code=500, detail="BERT model or Document Clusterer not initialized for this dataset.")
    
    original_query = request.query
    processed_query = original_query

    if request.apply_spell_correction and global_preprocessor:
        corrected_query = global_preprocessor.correct_query_spelling(original_query)
        if corrected_query.lower() != original_query.lower():
            processed_query = corrected_query
            print(f"Spell Correction Applied: '{original_query}' -> '{corrected_query}'")
        else:
            print(f"Spell Correction: No changes to '{original_query}'")
    
    print(f"\n--- Cluster Search Request ---")
    print(f"Query: '{processed_query}'")
    print(f"Target Cluster: {request.target_cluster_id if request.target_cluster_id is not None else 'Auto-detect'}")
    print(f"Top K: {request.top_k}")

    query_embedding = dataset_info.bert_model.encode_query(processed_query)

    target_cluster_id = request.target_cluster_id
    if target_cluster_id is None:
        nearest_cluster_id = dataset_info.document_clusterer.find_nearest_cluster(query_embedding)
        target_cluster_id = nearest_cluster_id
        print(f"Query assigned to cluster: {target_cluster_id}")
    else:
        print(f"Searching explicitly within cluster: {target_cluster_id}")

    cluster_doc_ids = dataset_info.document_clusterer.get_documents_in_cluster(target_cluster_id)
    if not cluster_doc_ids:
        return []

    start_time_cluster_search = time.time()
    
    if dataset_info.bert_model.document_embeddings_matrix is None:
        raise HTTPException(status_code=500, detail="BERT model embeddings not available for clustering search.")
    
    cluster_embeddings_map = {
        doc_id: dataset_info.bert_model.document_embeddings_matrix[
            dataset_info.bert_model.doc_id_map[doc_id]
        ]
        for doc_id in cluster_doc_ids if doc_id in dataset_info.bert_model.doc_id_map
    }
    
    cluster_raw_documents_dict = {
        doc_id: dataset_info.raw_documents_dict[doc_id]
        for doc_id in cluster_doc_ids if doc_id in dataset_info.raw_documents_dict
    }

    original_embeddings = dataset_info.bert_model.document_embeddings_matrix
    original_doc_id_map = dataset_info.bert_model.doc_id_map
    original_reverse_doc_id_map = dataset_info.bert_model.reverse_doc_id_map
    original_documents_text = dataset_info.bert_model.documents_text

    temp_doc_ids_list = list(cluster_embeddings_map.keys())
    if not temp_doc_ids_list:
        return []
        
    temp_embeddings_matrix = np.array([cluster_embeddings_map[doc_id] for doc_id in temp_doc_ids_list])
    temp_doc_id_map = {doc_id: i for i, doc_id in enumerate(temp_doc_ids_list)}
    temp_reverse_doc_id_map = {i: doc_id for i, doc_id in enumerate(temp_doc_ids_list)}

    dataset_info.bert_model.document_embeddings_matrix = temp_embeddings_matrix
    dataset_info.bert_model.doc_id_map = temp_doc_id_map
    dataset_info.bert_model.reverse_doc_id_map = temp_reverse_doc_id_map
    dataset_info.bert_model.documents_text = cluster_raw_documents_dict

    results = dataset_info.bert_model.search(processed_query, top_k=request.top_k, encode_query_again=False)
    
    dataset_info.bert_model.document_embeddings_matrix = original_embeddings
    dataset_info.bert_model.doc_id_map = original_doc_id_map
    dataset_info.bert_model.reverse_doc_id_map = original_reverse_doc_id_map
    dataset_info.bert_model.documents_text = original_documents_text

    end_time_cluster_search = time.time()
    print(f"Cluster search took {end_time_cluster_search - start_time_cluster_search:.4f} seconds.")

    response_results = []
    for doc_id, score in results:
        original_text = dataset_info.raw_documents_dict.get(doc_id)
        if original_text is None:
            if isinstance(doc_id, str) and doc_id.isdigit():
                original_text = dataset_info.raw_documents_dict.get(int(doc_id))
            elif isinstance(doc_id, int):
                original_text = dataset_info.raw_documents_dict.get(str(doc_id))

        text_preview = (original_text[:200] + '...') if original_text else "Text not available."
        response_results.append(SearchResult(doc_id=doc_id, score=score, text_preview=text_preview))
    
    return response_results

@app.post("/evaluate/{dataset_name}", response_model=Dict[str, EvaluationMetrics])
async def evaluate(dataset_name: str, request: EvaluationRequest):
    dataset_info = global_datasets_cache.get(dataset_name)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")
    
    if not dataset_info.queries or not dataset_info.qrels:
        raise HTTPException(status_code=400, detail="Queries or Qrels data not loaded for this dataset for evaluation.")

    models_to_evaluate = {
        'TF-IDF': dataset_info.vsm_model,
        'BERT': dataset_info.bert_model,
        'Hybrid': dataset_info.hybrid_model
    }
    
    queries_for_eval = {q.query_id: q for q in dataset_info.queries} 

    qrels_for_eval = defaultdict(dict)
    for qrel_item in dataset_info.qrels:
        qrels_for_eval[qrel_item.query_id][qrel_item.doc_id] = qrel_item.relevance
    
    print(f"Starting evaluation for dataset: {dataset_name}...")
    print(f"Evaluation options: Clustering={request.use_clustering_for_bert_evaluation}, PRF={request.use_prf_for_evaluation}")
    if request.use_prf_for_evaluation:
        print(f"  PRF Initial Model: {request.prf_initial_model}, Top N Docs: {request.prf_top_n_docs}, Expansion Terms: {request.prf_num_expansion_terms}, Final Model: {request.prf_final_model or request.prf_initial_model}")

    if global_preprocessor is None:
        raise HTTPException(status_code=500, detail="TextPreprocessor not initialized.")
    if global_query_optimizer is None:
        raise HTTPException(status_code=500, detail="QueryOptimizer not initialized.")

    evaluation_results = evaluate_models(
        models=models_to_evaluate, 
        queries=queries_for_eval, 
        qrels=qrels_for_eval, 
        k_values=[10],
        raw_documents_dict=dataset_info.raw_documents_dict,
        preprocessor=global_preprocessor,
        query_optimizer=global_query_optimizer,
        document_clusterer=dataset_info.document_clusterer,
        use_clustering_for_bert=request.use_clustering_for_bert_evaluation,
        use_prf=request.use_prf_for_evaluation,
        prf_initial_model_name=request.prf_initial_model,
        prf_top_n_docs=request.prf_top_n_docs,
        prf_num_expansion_terms=request.prf_num_expansion_terms,
        prf_final_model_name=request.prf_final_model
    )
    print(f"Evaluation for dataset {dataset_name} completed.")

    formatted_results = {}
    for model_name, metrics in evaluation_results.items():
        formatted_results[model_name] = EvaluationMetrics(
            model_name=model_name,
            map=metrics.get('map', 0.0), 
            recall=metrics.get('recall', 0.0),
            precision_at_10=metrics.get('precision_at_10', 0.0),
            mrr=metrics.get('mrr', 0.0),
        )
    return formatted_results