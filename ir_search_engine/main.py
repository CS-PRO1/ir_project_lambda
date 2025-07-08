import os
import sys
from collections import namedtuple
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

# Define paths for saving/loading indices and embeddings
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
INDEXES_DIR = os.path.join(DATA_DIR, 'indexes')
os.makedirs(INDEXES_DIR, exist_ok=True) # Ensure the indexes directory exists

# Named tuple for convenience to store dataset info
DatasetInfo = namedtuple('DatasetInfo', ['name', 'docs', 'queries', 'qrels', 'inverted_index', 'bert_embeddings', 'vsm_model', 'bert_model', 'hybrid_model'])

def load_or_build_inverted_index(dataset_name, docs, preprocessor):
    """Loads an existing InvertedIndex or builds it from scratch."""
    index_filepath = os.path.join(INDEXES_DIR, f"{dataset_name}_inverted_index.json")
    
    index = InvertedIndex()
    if os.path.exists(index_filepath):
        print(f"Loading Inverted Index for {dataset_name} from {index_filepath}...")
        index.load(index_filepath)
        print(f"Inverted index loaded from {index_filepath}")
    else:
        print(f"Inverted Index not found for {dataset_name}. Building...")
        # Preprocess documents for VSM (using lemmatization, no n-grams for base VSM)
        # We need a list of texts for preprocessor.preprocess_documents
        doc_texts_list = [doc.text for doc in docs]
        
        # Add tqdm to preprocessing step for VSM
        print("Preprocessing documents for TF-IDF...")
        preprocessed_docs_list = preprocessor.preprocess_documents(
            doc_texts_list, use_stemming=False, use_lemmatization=True, add_ngrams=False,
            desc="Preprocessing documents" # desc for tqdm
        )
        
        # InvertedIndex expects {doc_id: preprocessed_text}
        # Create a mapping from original doc_ids to preprocessed texts
        preprocessed_docs_dict = {
            docs[i].doc_id: preprocessed_docs_list[i] 
            for i in tqdm(range(len(docs)), desc="Mapping preprocessed docs")
        }

        print("Building Inverted Index...")
        index.build_index(preprocessed_docs_dict)
        index.save(index_filepath)
        print(f"Inverted index built and saved to {index_filepath}")
    return index

def load_or_index_bert_embeddings(dataset_name, raw_documents_dict):
    """Loads existing BERT embeddings or generates them from scratch."""
    embeddings_filepath_base = os.path.join(INDEXES_DIR, f"{dataset_name}_bert_embeddings")
    
    bert_model = BERTRetrievalModel() # Initialize model (downloads if not present)
    
    # Check for all necessary files for loading
    if os.path.exists(f"{embeddings_filepath_base}.npy") and \
       os.path.exists(f"{embeddings_filepath_base}_map.json") and \
       os.path.exists(f"{embeddings_filepath_base}_text.json"): # Check for _text.json too
        print(f"Loading BERT embeddings for {dataset_name} from {embeddings_filepath_base}.npy...")
        bert_model.load_embeddings(embeddings_filepath_base)
    else:
        print(f"BERT embeddings or associated files not found for {dataset_name}. Generating...")
        bert_model.index_documents(raw_documents_dict) # This method has its own tqdm
        bert_model.save_embeddings(embeddings_filepath_base)
    
    # ENSURE documents_text is always populated from the fresh raw_documents_dict
    # This acts as a fallback/reinforcement for consistency, especially after loading
    # as load_embeddings will populate it from its own saved file.
    bert_model.documents_text = raw_documents_dict 
    
    return bert_model

def get_raw_documents_dict(docs_list):
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
            print(f"  {key}. {name}")
        choice = input("Enter your choice (1 or 2): ").strip()
        selected_dataset_name = available_datasets.get(choice)
        if not selected_dataset_name:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"\nLoading data for {selected_dataset_name}...")
    docs, queries, qrels = None, None, None # Initialize to None
    if selected_dataset_name == 'antique_train':
        docs, queries, qrels = data_loader.load_antique_train()
    elif selected_dataset_name == 'beir_webist_touche2020':
        docs, queries, qrels = data_loader.load_beir_webist_touche2020()
    
    if docs is None: # Should not happen with valid choice, but good for safety
        print("Failed to load documents. Exiting.")
        return

    raw_documents_dict = get_raw_documents_dict(docs)

    # --- Indexing and Embedding Generation ---
    print("\nPreparing retrieval models (this may take a while for the first run)...")
    
    # Inverted Index for VSM
    inverted_index = load_or_build_inverted_index(selected_dataset_name, docs, preprocessor)
    vsm_model = VectorSpaceModel(inverted_index, preprocessor)
    print(f"VSM model ready with {vsm_model.total_documents} documents.")

    # BERT Embeddings
    bert_model = load_or_index_bert_embeddings(selected_dataset_name, raw_documents_dict)
    print(f"BERT model ready with {len(bert_model.doc_id_map)} embeddings.")

    # Hybrid Ranker
    hybrid_ranker = HybridRanker(inverted_index, preprocessor, bert_model)
    print("Hybrid Ranker ready.")

    current_dataset_info = DatasetInfo(
        name=selected_dataset_name,
        docs=docs, # Raw docs list
        queries=queries,
        qrels=qrels,
        inverted_index=inverted_index,
        bert_embeddings=bert_model.document_embeddings_matrix, # Store matrix for reference
        vsm_model=vsm_model,
        bert_model=bert_model,
        hybrid_model=hybrid_ranker
    )

    # --- Interactive Search Loop ---
    while True:
        print("\n--- Search Options ---")
        print("1. TF-IDF (Vector Space Model)")
        print("2. BERT Embeddings (Semantic Search)")
        print("3. Hybrid Search (TF-IDF + BERT)")
        print("4. Change Dataset")
        print("5. Exit")
        
        model_choice = input("Select a retrieval model (1-5): ").strip()

        if model_choice == '5':
            print("Exiting search engine. Goodbye!")
            break
        
        if model_choice == '4':
            print("Changing dataset...")
            # Recursively call main to allow dataset selection again
            # This will re-initialize everything for the new dataset
            main() 
            return # Exit current main call to prevent double loop

        if model_choice not in ['1', '2', '3']:
            print("Invalid model choice. Please enter 1, 2, 3, 4, or 5.")
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
            # Hybrid search can take additional kwargs for VSM, e.g., add_ngrams=True
            results = current_dataset_info.hybrid_model.hybrid_search(query_text, top_k=10, fusion_method='rrf') # Default to RRF

        end_time = time.time()
        
        print(f"\n--- Results ({model_used} - {len(results)} found in {end_time - start_time:.4f} seconds) ---")
        if results:
            for i, (doc_id, score) in enumerate(results):
                # --- START DEBUG PRINTS FOR TEXT RETRIEVAL ---
                # 1. Check the type of doc_id coming from the search result
                # print(f"DEBUG: Result Doc ID: {doc_id}, Type: {type(doc_id)}")
                
                # Retrieve original text using the bert_model's documents_text, which stores all raw texts
                # This dict should have consistent types from raw_documents_dict
                original_text = current_dataset_info.bert_model.documents_text.get(doc_id)

                # DEBUG: If original_text is still None, it means the key was not found.
                # Let's try converting types as a fallback for debugging.
                if original_text is None:
                    # print(f"DEBUG: Doc ID '{doc_id}' NOT found directly. Attempting type conversion...")
                    converted_doc_id = None
                    if isinstance(doc_id, str) and doc_id.isdigit():
                        converted_doc_id = int(doc_id)
                    elif isinstance(doc_id, int):
                        converted_doc_id = str(doc_id)
                    
                    if converted_doc_id is not None:
                        # print(f"DEBUG: Trying converted Doc ID '{converted_doc_id}', Type: {type(converted_doc_id)}")
                        original_text = current_dataset_info.bert_model.documents_text.get(converted_doc_id)
                        # if original_text is not None:
                            # print(f"DEBUG: Found text with converted ID: '{original_text[:50]}'")
                        # else:
                            # print("DEBUG: Still no text found after conversion attempt.")

                if original_text is None:
                    original_text = "Original text not available (ID mismatch or not found)."
                # --- END DEBUG PRINTS FOR TEXT RETRIEVAL ---

                print(f"{i+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                print(f"   Text: {original_text[:200]}...") # Print first 200 chars
        else:
            print("No relevant documents found for your query.")

if __name__ == "__main__":
    main()