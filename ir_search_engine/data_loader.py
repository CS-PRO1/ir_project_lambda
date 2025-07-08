import ir_datasets
from collections import namedtuple
import threading
from tqdm import tqdm
import os
import json # For saving documents, queries, qrels as JSON lines

# Define named tuples for clarity and consistency
Document = namedtuple('Document', ['doc_id', 'text'])
Query = namedtuple('Query', ['query_id', 'text'])
Qrel = namedtuple('Qrel', ['query_id', 'doc_id', 'relevance', 'iteration'])

class DataLoader:
    def __init__(self, base_data_path='../data'): # Relative path to data folder from ir_search_engine
        """
        Initializes the DataLoader.
        Args:
            base_data_path (str): The base directory to save/load data.
        """
        self.datasets = {}
        self.base_data_path = os.path.abspath(base_data_path) # Get absolute path
        os.makedirs(self.base_data_path, exist_ok=True) # Ensure base data directory exists

    def _get_dataset_local_path(self, dataset_name):
        """Returns the local path where a specific dataset's files should be stored."""
        return os.path.join(self.base_data_path, dataset_name.replace('/', '_')) # e.g., 'antique_train'

    def _save_data_to_disk(self, data_list, filepath, data_type_name):
        """Saves a list of namedtuple objects to disk as JSON lines."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
        print(f"Saving {len(data_list)} {data_type_name} to {filepath}...")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in tqdm(data_list, desc=f"Saving {data_type_name}", unit="items"):
                    f.write(json.dumps(item._asdict()) + '\n') # Convert namedtuple to dict and save as JSON line
            print(f"Successfully saved {data_type_name}.")
        except Exception as e:
            print(f"Error saving {data_type_name} to {filepath}: {e}")

    def _load_data_from_disk(self, filepath, namedtuple_type, data_type_name):
        """Loads data from disk (JSON lines) into a list of namedtuple objects."""
        if not os.path.exists(filepath):
            return None
        print(f"Loading {data_type_name} from {filepath}...")
        data_list = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Loading {data_type_name}", unit="items"):
                    data_dict = json.loads(line.strip())
                    data_list.append(namedtuple_type(**data_dict))
            print(f"Successfully loaded {len(data_list)} {data_type_name}.")
            return data_list
        except Exception as e:
            print(f"Error loading {data_type_name} from {filepath}: {e}")
            return None

    def _load_dataset_part(self, dataset_name, part_type, output_list, lock):
        """
        Helper method to load a specific part (docs, queries, or qrels) of a dataset.
        First tries to load from local disk; if not found, loads from ir_datasets.
        """
        local_path = os.path.join(self._get_dataset_local_path(dataset_name), f"{part_type}.jsonl")
        
        loaded_data = self._load_data_from_disk(local_path, 
                                                Document if part_type == 'docs' else (Query if part_type == 'queries' else Qrel),
                                                part_type)
        
        if loaded_data is not None:
            with lock:
                output_list.extend(loaded_data)
            return

        # If not found on disk, load from ir_datasets
        print(f"'{part_type}.jsonl' not found at {local_path}. Loading from ir_datasets for '{dataset_name}'...")
        try:
            dataset = ir_datasets.load(dataset_name)
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            print("Please ensure 'ir_datasets' is configured correctly and datasets are accessible.")
            return

        iterator = None
        total = 0
        get_item = None

        if part_type == 'docs':
            iterator = dataset.docs_iter()
            total = dataset.docs_count()
            get_item = lambda item: Document(item.doc_id, item.text)
        elif part_type == 'queries':
            iterator = dataset.queries_iter()
            total = dataset.queries_count()
            get_item = lambda item: Query(item.query_id, item.text)
        elif part_type == 'qrels':
            iterator = dataset.qrels_iter()
            total = dataset.qrels_count()
            get_item = lambda item: Qrel(item.query_id, item.doc_id, item.relevance, item.iteration)
        else:
            raise ValueError(f"Unknown part_type: {part_type}")

        if iterator and get_item:
            local_data = []
            for item in tqdm(iterator, total=total, desc=f"Loading {dataset_name} {part_type}", unit="items"):
                local_data.append(get_item(item))
            with lock:
                output_list.extend(local_data)
            
            # Save to disk after successful loading from ir_datasets
            self._save_data_to_disk(local_data, local_path, part_type)

    def load_dataset_multithreaded(self, dataset_name):
        """
        Loads documents, queries, and qrels for a given dataset concurrently using threads.
        Prioritizes loading from local disk. If not found, loads from ir_datasets and saves.
        """
        docs = []
        queries = []
        qrels = []

        doc_lock = threading.Lock()
        query_lock = threading.Lock()
        qrel_lock = threading.Lock()

        threads = []
        threads.append(threading.Thread(target=self._load_dataset_part, args=(dataset_name, 'docs', docs, doc_lock)))
        threads.append(threading.Thread(target=self._load_dataset_part, args=(dataset_name, 'queries', queries, query_lock)))
        threads.append(threading.Thread(target=self._load_dataset_part, args=(dataset_name, 'qrels', qrels, qrel_lock)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return docs, queries, qrels

    def load_antique_train(self):
        """
        Loads the 'antique/train' dataset.
        """
        print("Loading antique/train dataset...")
        docs, queries, qrels = self.load_dataset_multithreaded('antique/train')
        self.datasets['antique_train'] = {'docs': docs, 'queries': queries, 'qrels': qrels}
        print(f"Loaded antique/train: {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels.")
        return docs, queries, qrels

    def load_beir_webist_touche2020(self):
        """
        Loads the 'beir/webis-touche2020' dataset.
        """
        print("Loading beir/webis-touche2020 dataset...")
        docs, queries, qrels = self.load_dataset_multithreaded('beir/webis-touche2020')
        self.datasets['beir_webist_touche2020'] = {'docs': docs, 'queries': queries, 'qrels': qrels}
        print(f"Loaded beir/webis-touche2020: {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels.")
        return docs, queries, qrels

    def get_dataset(self, name):
        """
        Retrieves a previously loaded dataset by name.
        """
        return self.datasets.get(name)

# --- Test / Example Usage (for immediate testing of this module) ---
if __name__ == "__main__":
    print("--- Testing data_loader.py with local file saving ---")
    
    # Initialize DataLoader with a path to your project's 'data' folder
    # Assuming this script is in ir_search_engine/, so '../data' refers to the parent's 'data' folder
    loader = DataLoader(base_data_path='../data')

    # Test loading antique/train
    print("\nAttempting to load antique/train...")
    antique_docs, antique_queries, antique_qrels = loader.load_antique_train()

    if antique_docs and antique_queries and antique_qrels:
        print(f"\nSuccessfully loaded antique/train. Sample data:")
        print(f"First document: {antique_docs[0]}")
        print(f"First query: {antique_queries[0]}")
        print(f"First qrel: {antique_qrels[0]}")
        print(f"Total antique/train documents: {len(antique_docs)}")
        print(f"Total antique/train queries: {len(antique_queries)}")
        print(f"Total antique/train qrels: {len(antique_qrels)}")
    else:
        print("\nFailed to load antique/train data or data is empty.")

    # Test loading beir/webis-touche2020
    print("\nAttempting to load beir/webis-touche2020...")
    webist_docs, webist_queries, webist_qrels = loader.load_beir_webist_touche2020()

    if webist_docs and webist_queries and webist_qrels:
        print(f"\nSuccessfully loaded beir/webis-touche2020. Sample data:")
        print(f"First document: {webist_docs[0]}")
        print(f"First query: {webist_queries[0]}")
        print(f"First qrel: {webist_qrels[0]}")
        print(f"Total beir/webis-touche2020 documents: {len(webist_docs)}")
        print(f"Total beir/webis-touche2020 queries: {len(webist_queries)}")
        print(f"Total beir/webis-touche2020 qrels: {len(webist_qrels)}")
    else:
        print("\nFailed to load beir/webis-touche2020 data or data is empty.")

    print("\n--- data_loader.py testing complete ---")