import os
import sys
from collections import namedtuple, defaultdict
import time
from tqdm import tqdm # Import tqdm for progress bars

# Ensure current directory is in PATH for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all custom modules
from data_loader import DataLoader, Document, Query, Qrel
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from retrieval_model import VectorSpaceModel
from bert_retrieval import BERTRetrievalModel
from hybrid_retrieval import HybridRanker
from query_optimizer import QueryOptimizer # NEW: Import QueryOptimizer

# Import evaluation functions
from evaluator import evaluate_models 

# Import type hints
from typing import Dict, List, Union

# Define paths for saving/loading indices and embeddings
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
INDEXES_DIR = os.path.join(DATA_DIR, 'indexes')
os.makedirs(INDEXES_DIR, exist_ok=True) # Ensure the indexes directory exists

# Named tuple for convenience to store dataset info
DatasetInfo = namedtuple('DatasetInfo', ['name', 'docs', 'queries', 'qrels', 'inverted_index', 'bert_embeddings', 'vsm_model', 'bert_model', 'hybrid_model'])

def load_or_build_inverted_index(dataset_name: str, docs: List[Document], preprocessor: TextPreprocessor) -> InvertedIndex:
    """Loads an existing InvertedIndex or builds it from scratch."""
    index_filepath = os.path.join(INDEXES_DIR, f"{dataset_name}_inverted_index.json")
    
    index = InvertedIndex()
    if os.path.exists(index_filepath):
        print(f"Loading Inverted Index for {dataset_name} from {index_filepath}...")
        try:
            index.load(index_filepath)
            print(f"Inverted index loaded from {index_filepath}")
        except Exception as e:
            print(f"Error loading inverted index: {e}. Rebuilding...")
            if os.path.exists(index_filepath):
                os.remove(index_filepath)
            pass
    
    if not (hasattr(index, 'index') and index.index and len(index.index) > 0 and \
            hasattr(index, 'document_lengths') and index.document_lengths): 
        print(f"Inverted Index not found or empty for {dataset_name}. Building...")
        doc_texts_list = [doc.text for doc in docs]
        
        print("Preprocessing documents for TF-IDF...")
        preprocessed_docs_list = preprocessor.preprocess_documents(
            doc_texts_list, use_stemming=False, use_lemmatization=True, add_ngrams=False,
            desc="Preprocessing documents"
        )
        
        preprocessed_docs_dict = {
            docs[i].doc_id: preprocessed_docs_list[i] 
            for i in tqdm(range(len(docs)), desc="Mapping preprocessed docs")
        }

        print("Building Inverted Index...")
        index.build_index(preprocessed_docs_dict)
        index.save(index_filepath)
        print(f"Inverted index built and saved to {index_filepath}")
    return index

def load_or_index_bert_embeddings(dataset_name: str, raw_documents_dict: Dict[Union[int, str], str]) -> BERTRetrievalModel:
    """Loads existing BERT embeddings or generates them from scratch."""
    embeddings_filepath_base = os.path.join(INDEXES_DIR, f"{dataset_name}_bert_embeddings")
    
    bert_model = BERTRetrievalModel()
    
    bert_embeddings_npy = f"{embeddings_filepath_base}.npy"
    bert_map_json = f"{embeddings_filepath_base}_map.json"
    bert_text_json = f"{embeddings_filepath_base}_text.json"

    if os.path.exists(bert_embeddings_npy) and \
       os.path.exists(bert_map_json) and \
       os.path.exists(bert_text_json):
        print(f"Loading BERT embeddings for {dataset_name} from {embeddings_filepath_base}.npy...")
        try:
            bert_model.load_embeddings(embeddings_filepath_base)
            print("BERT embeddings loaded.")
            if not bert_model.documents_text:
                bert_model.documents_text = raw_documents_dict
            print(f"BERT model ready with {len(bert_model.doc_id_map)} embeddings.")
            return bert_model
        except Exception as e:
            print(f"Error loading BERT embeddings: {e}. Rebuilding...")
            for f in [bert_embeddings_npy, bert_map_json, bert_text_json]:
                if os.path.exists(f):
                    os.remove(f)
            pass
    
    print(f"BERT embeddings or associated files not found/corrupt for {dataset_name}. Generating...")
    bert_model.index_documents(raw_documents_dict)
    bert_model.save_embeddings(embeddings_filepath_base)
    print(f"BERT embeddings generated and saved. BERT model ready with {len(bert_model.doc_id_map)} embeddings.")
    
    bert_model.documents_text = raw_documents_dict 
    
    return bert_model

def get_raw_documents_dict(docs_list: List[Document]) -> Dict[Union[int, str], str]:
    """Converts a list of Document namedtuples to a dict {doc_id: text}."""
    return {doc.doc_id: doc.text for doc in tqdm(docs_list, desc="Preparing raw document dict")}

def main():
    print("--- Information Retrieval Search Engine ---")

    data_loader = DataLoader(base_data_path=DATA_DIR)
    preprocessor = TextPreprocessor(language='english')

    # --- Dataset Selection ---
    available_datasets = {
        '1': 'antique_train',
        '2': 'beir_webist_touche2020'
    }
    
    selected_dataset_name = None
    while selected_dataset_name not in available_datasets.values():
        print("\nSelect a dataset to load:")
        for key, name in available_datasets.items():
            print(f"   {key}. {name}")
        choice = input("Enter your choice (1 or 2): ").strip()
        selected_dataset_name = available_datasets.get(choice)
        if not selected_dataset_name:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"\nLoading data for {selected_dataset_name}...")
    docs, queries, qrels = None, None, None
    if selected_dataset_name == 'antique_train':
        docs, queries, qrels = data_loader.load_antique_train()
    elif selected_dataset_name == 'beir_webist_touche2020':
        docs, queries, qrels = data_loader.load_beir_webist_touche2020()
    
    if docs is None:
        print("Failed to load documents. Exiting.")
        return

    raw_documents_dict = get_raw_documents_dict(docs)

    # --- Indexing and Embedding Generation ---
    print("\nPreparing retrieval models (this may take a while for the first run)...")
    
    inverted_index = load_or_build_inverted_index(selected_dataset_name, docs, preprocessor)
    vsm_model = VectorSpaceModel(inverted_index, preprocessor)
    print(f"VSM model ready with {vsm_model.total_documents} documents.")

    bert_model = load_or_index_bert_embeddings(selected_dataset_name, raw_documents_dict)
    print(f"BERT model ready with {len(bert_model.doc_id_map)} embeddings.")

    hybrid_ranker = HybridRanker(inverted_index, preprocessor, bert_model)
    print("Hybrid Ranker ready.")

    current_dataset_info = DatasetInfo(
        name=selected_dataset_name,
        docs=docs,
        queries=queries,
        qrels=qrels,
        inverted_index=inverted_index,
        bert_embeddings=bert_model.document_embeddings_matrix,
        vsm_model=vsm_model,
        bert_model=bert_model,
        hybrid_model=hybrid_ranker
    )

    # NEW: Initialize QueryOptimizer
    query_optimizer = QueryOptimizer(preprocessor, vsm_model, bert_model)


    # --- Interactive Search Loop ---
    while True:
        print("\n--- Search Options ---")
        print("1. TF-IDF (Vector Space Model)")
        print("2. BERT Embeddings (Semantic Search)")
        print("3. Hybrid Search (TF-IDF + BERT)")
        print("4. Change Dataset")
        print("5. Run Evaluation") 
        print("6. Exit")
        print("7. Query Optimization (Pseudo-Relevance Feedback)") # NEW OPTION
        
        model_choice = input("Select an option (1-7): ").strip()

        if model_choice == '6': 
            print("Exiting search engine. Goodbye!")
            break
        
        if model_choice == '4':
            print("Changing dataset...")
            main() 
            return

        if model_choice == '5': 
            if not current_dataset_info.queries or not current_dataset_info.qrels:
                print("Cannot run evaluation: Queries or Qrels data not loaded for this dataset.")
                continue
            
            models_to_evaluate = {
                'TF-IDF': current_dataset_info.vsm_model,
                'BERT': current_dataset_info.bert_model,
                'Hybrid': current_dataset_info.hybrid_model
            }
            
            eval_k_values = [1, 5, 10, 20]

            queries_for_eval = {q.query_id: q for q in current_dataset_info.queries} 

            qrels_for_eval = defaultdict(dict)
            for qrel_item in current_dataset_info.qrels:
                qrels_for_eval[qrel_item.query_id][qrel_item.doc_id] = qrel_item.relevance
            
            evaluate_models(models_to_evaluate, queries_for_eval, qrels_for_eval, eval_k_values)
            continue
        
        # NEW: Handle Query Optimization option
        if model_choice == '7':
            original_query = input("Enter the query to optimize: ").strip()
            if not original_query:
                print("Query cannot be empty. Please try again.")
                continue

            print("Select initial search model for PRF:")
            print("  1. TF-IDF")
            print("  2. BERT")
            prf_model_choice = input("Enter choice (1 or 2): ").strip()
            
            retrieval_model_name_for_prf = ""
            if prf_model_choice == '1':
                retrieval_model_name_for_prf = 'TF-IDF'
            elif prf_model_choice == '2':
                retrieval_model_name_for_prf = 'BERT'
            else:
                print("Invalid model choice for PRF. Aborting optimization.")
                continue

            try:
                top_n_docs = int(input("Number of top documents for feedback (e.g., 5, 10, 20 - default 5): ") or 5)
                num_terms = int(input("Number of terms to add to query (e.g., 3, 5 - default 3): ") or 3)
            except ValueError:
                print("Invalid number entered. Using default values (5 docs, 3 terms).")
                top_n_docs = 5
                num_terms = 3

            expanded_query = query_optimizer.expand_query_with_prf(
                original_query,
                retrieval_model_name_for_prf,
                top_n_docs_for_prf=top_n_docs,
                num_expansion_terms=num_terms
            )
            
            if expanded_query != original_query:
                perform_expanded_search = input("Do you want to perform a search with the expanded query? (yes/no): ").strip().lower()
                if perform_expanded_search == 'yes':
                    # Directly perform search with expanded query using the previous model_choice context
                    # Or, better, ask user which model to use for the *expanded* query
                    print("\nWhich model to use for the expanded query search?")
                    print("  1. TF-IDF")
                    print("  2. BERT")
                    print("  3. Hybrid")
                    expanded_search_model_choice = input("Enter choice (1-3): ").strip()

                    search_results = []
                    search_model_name = ""
                    if expanded_search_model_choice == '1':
                        search_model_name = "TF-IDF"
                        search_results = current_dataset_info.vsm_model.search(expanded_query, top_k=10)
                    elif expanded_search_model_choice == '2':
                        search_model_name = "BERT Embeddings"
                        search_results = current_dataset_info.bert_model.search(expanded_query, top_k=10)
                    elif expanded_search_model_choice == '3':
                        search_model_name = "Hybrid Search"
                        search_results = current_dataset_info.hybrid_model.hybrid_search(
                            expanded_query, 
                            top_k=10, 
                            vsm_weight=0.1, bert_weight=0.9, top_k_bert_initial=200
                        )
                    else:
                        print("Invalid choice. Returning to main menu.")
                        continue # Go back to main menu
                    
                    print(f"\n--- Results ({search_model_name} with expanded query - {len(search_results)} found) ---")
                    if search_results:
                        for i, (doc_id, score) in enumerate(search_results):
                            original_text = current_dataset_info.bert_model.documents_text.get(doc_id)
                            if original_text is None:
                                converted_doc_id = None
                                if isinstance(doc_id, str) and doc_id.isdigit(): converted_doc_id = int(doc_id)
                                elif isinstance(doc_id, int): converted_doc_id = str(doc_id)
                                if converted_doc_id is not None:
                                    original_text = current_dataset_info.bert_model.documents_text.get(converted_doc_id)
                            if original_text is None:
                                original_text = "Original text not available."
                            
                            print(f"{i+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                            print(f"   Text: {original_text[:200]}...")
                    else:
                        print("No relevant documents found with the expanded query.")
                else:
                    print("Expanded query generated, but no search performed. Returning to main menu.")
            else:
                print("Query not expanded. Returning to main menu.")
            continue # Go back to search options menu
        
        # Original search options (1, 2, 3)
        if model_choice not in ['1', '2', '3']: 
            print("Invalid model choice. Please enter 1, 2, 3, 4, 5, 6 or 7.")
            continue

        query_text = input("Enter your query: ").strip()
        if not query_text:
            print("Query cannot be empty. Please try again.")
            continue

        start_time = time.time()
        results = []
        model_used = ""

        if model_choice == '1':
            model_used = "TF-IDF"
            results = current_dataset_info.vsm_model.search(query_text, top_k=10)
        elif model_choice == '2':
            model_used = "BERT Embeddings"
            results = current_dataset_info.bert_model.search(query_text, top_k=10)
        elif model_choice == '3':
            model_used = "Hybrid Search"
            results = current_dataset_info.hybrid_model.hybrid_search(
                query_text, 
                top_k=10, 
                vsm_weight=0.1,                
                bert_weight=0.9,               
                top_k_bert_initial=200         
            )

        end_time = time.time()
        
        print(f"\n--- Results ({model_used} - {len(results)} found in {end_time - start_time:.4f} seconds) ---")
        if results:
            for i, (doc_id, score) in enumerate(results):
                original_text = current_dataset_info.bert_model.documents_text.get(doc_id)

                if original_text is None:
                    converted_doc_id = None
                    if isinstance(doc_id, str) and doc_id.isdigit():
                        converted_doc_id = int(doc_id)
                    elif isinstance(doc_id, int):
                        converted_doc_id = str(doc_id)
                    
                    if converted_doc_id is not None:
                        original_text = current_dataset_info.bert_model.documents_text.get(converted_doc_id)

                if original_text is None:
                    original_text = "Original text not available (ID mismatch or not found)."
                
                print(f"{i+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                print(f"   Text: {original_text[:200]}...")
        else:
            print("No relevant documents found for your query.")

if __name__ == "__main__":
    main()