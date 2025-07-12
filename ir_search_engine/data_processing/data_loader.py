import ir_datasets
from collections import namedtuple
import threading
from tqdm import tqdm
import os
import json

Document = namedtuple('Document', ['doc_id', 'text'])
Query = namedtuple('Query', ['query_id', 'text'])
Qrel = namedtuple('Qrel', ['query_id', 'doc_id', 'relevance', 'iteration'])

class DataLoader:
    def __init__(self, base_data_path='../data'):
        self.datasets = {}
        self.base_data_path = os.path.abspath(base_data_path)
        os.makedirs(self.base_data_path, exist_ok=True)

    def _get_dataset_local_path(self, dataset_name):
        return os.path.join(self.base_data_path, dataset_name.replace('/', '_'))

    def _save_data_to_disk(self, data_list, filepath, data_type_name):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving {len(data_list)} {data_type_name} to {filepath}...")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in tqdm(data_list, desc=f"Saving {data_type_name}", unit="items"):
                    f.write(json.dumps(item._asdict()) + '\n')
            print(f"Successfully saved {data_type_name}.")
        except Exception as e:
            print(f"Error saving {data_type_name} to {filepath}: {e}")

    def _load_data_from_disk(self, filepath, namedtuple_type, data_type_name):
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
        local_path = os.path.join(self._get_dataset_local_path(dataset_name), f"{part_type}.jsonl")
        
        loaded_data = self._load_data_from_disk(local_path, 
                                                Document if part_type == 'docs' else (Query if part_type == 'queries' else Qrel),
                                                part_type)
        
        if loaded_data is not None:
            with lock:
                output_list.extend(loaded_data)
            return

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
            
            self._save_data_to_disk(local_data, local_path, part_type)

    def load_dataset_multithreaded(self, dataset_name):
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
        print("Loading antique/train dataset...")
        docs, queries, qrels = self.load_dataset_multithreaded('antique/train')
        self.datasets['antique_train'] = {'docs': docs, 'queries': queries, 'qrels': qrels}
        print(f"Loaded antique/train: {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels.")
        return docs, queries, qrels

    def load_beir_webist_touche2020(self):
        print("Loading beir/webis-touche2020 dataset...")
        docs, queries, qrels = self.load_dataset_multithreaded('beir/webis-touche2020')
        self.datasets['beir_webist_touche2020'] = {'docs': docs, 'queries': queries, 'qrels': qrels}
        print(f"Loaded beir/webis-touche2020: {len(docs)} docs, {len(queries)} queries, {len(qrels)} qrels.")
        return docs, queries, qrels

    def get_dataset(self, name):
        return self.datasets.get(name)