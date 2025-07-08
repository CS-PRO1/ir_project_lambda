import os
import json
import csv
from typing import Dict, Union, List, Tuple
from tqdm import tqdm # For progress bars when loading many files

class CorpusLoader:
    def __init__(self, corpus_path: str):
        """
        Initializes the CorpusLoader.

        :param corpus_path: The path to the corpus. Can be a directory for 'txt' files,
                            or a specific file path for 'json'/'csv' files.
        """
        self.corpus_path = corpus_path
        self.documents: Dict[Union[int, str], str] = {} # {doc_id: raw_text}
        self._next_auto_id = 0 # For assigning sequential integer IDs if not provided

    def _generate_next_id(self) -> int:
        """Generates a sequential integer ID."""
        _id = self._next_auto_id
        self._next_auto_id += 1
        return _id

    def load_documents(self, file_type: str = 'txt', **kwargs) -> Dict[Union[int, str], str]:
        """
        Loads documents from the specified corpus path based on the file_type.

        :param file_type: Type of files to load ('txt', 'json', 'csv').
                          'txt': Expects corpus_path to be a directory. Each .txt file is a document.
                                 Doc ID will be the filename (without extension) or a sequential integer.
                          'json': Expects corpus_path to be a .json file. Can be a list of objects
                                  or a single object containing 'docs' key (see example).
                          'csv': Expects corpus_path to be a .csv file.
        :param kwargs: Additional arguments specific to file type:
            - txt:
                - use_filename_as_id (bool): If True, use filename (without ext) as ID. Else, sequential int ID. (default: True)
            - json:
                - json_id_key (str): Key in JSON object for document ID. (default: 'id')
                - json_text_key (str): Key in JSON object for document text. (default: 'text')
                - json_is_list (bool): True if JSON file is a list of objects, False if a single object with a 'documents' key holding a list. (default: True)
            - csv:
                - csv_id_col (str): Column name in CSV for document ID. (default: 'id')
                - csv_text_col (str): Column name in CSV for document text. (default: 'text')
                - delimiter (str): CSV delimiter. (default: ',')
        :return: A dictionary {doc_id: raw_text}.
        :raises ValueError: If unsupported file_type or invalid path.
        """
        self.documents = {} # Reset documents for each load call
        self._next_auto_id = 0 # Reset auto ID counter

        print(f"Loading documents from '{self.corpus_path}' (type: {file_type})...")

        if file_type == 'txt':
            self._load_from_txt_directory(kwargs.get('use_filename_as_id', True))
        elif file_type == 'json':
            self._load_from_json_file(
                kwargs.get('json_id_key', 'id'),
                kwargs.get('json_text_key', 'text'),
                kwargs.get('json_is_list', True)
            )
        elif file_type == 'csv':
            self._load_from_csv_file(
                kwargs.get('csv_id_col', 'id'),
                kwargs.get('csv_text_col', 'text'),
                kwargs.get('delimiter', ',')
            )
        else:
            raise ValueError(f"Unsupported file_type: {file_type}. Supported types are 'txt', 'json', 'csv'.")
        
        print(f"Loaded {len(self.documents)} documents.")
        return self.documents

    def _load_from_txt_directory(self, use_filename_as_id: bool):
        """Loads documents from a directory of plain text files."""
        if not os.path.isdir(self.corpus_path):
            raise ValueError(f"Corpus path '{self.corpus_path}' is not a directory for 'txt' file_type.")
        
        filepaths = [os.path.join(self.corpus_path, f) for f in os.listdir(self.corpus_path) if f.endswith('.txt')]
        
        for filepath in tqdm(filepaths, desc="Loading .txt files"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
                    doc_id = os.path.splitext(os.path.basename(filepath))[0] if use_filename_as_id else self._generate_next_id()
                    self.documents[doc_id] = doc_text
            except Exception as e:
                print(f"Warning: Could not load file {filepath}: {e}")

    def _load_from_json_file(self, json_id_key: str, json_text_key: str, json_is_list: bool):
        """Loads documents from a JSON file."""
        if not os.path.isfile(self.corpus_path) or not self.corpus_path.endswith('.json'):
            raise ValueError(f"Corpus path '{self.corpus_path}' is not a .json file for 'json' file_type.")
        
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                documents_to_process = []
                if json_is_list:
                    if isinstance(data, list):
                        documents_to_process = data
                    else:
                        raise ValueError(f"JSON file expected to be a list of objects but found {type(data)}.")
                else: # Expects a single object with a 'documents' key holding a list
                    if isinstance(data, dict) and 'documents' in data and isinstance(data['documents'], list):
                        documents_to_process = data['documents']
                    else:
                        raise ValueError(f"JSON file expected to be an object with a 'documents' list, but format mismatch.")

                for item in tqdm(documents_to_process, desc="Loading .json entries"):
                    if json_id_key in item and json_text_key in item:
                        doc_id = item[json_id_key]
                        doc_text = item[json_text_key]
                        self.documents[doc_id] = doc_text
                    else:
                        print(f"Warning: JSON entry missing '{json_id_key}' or '{json_text_key}' keys: {item}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        except Exception as e:
            print(f"Warning: Could not load JSON file {self.corpus_path}: {e}")

    def _load_from_csv_file(self, csv_id_col: str, csv_text_col: str, delimiter: str):
        """Loads documents from a CSV file."""
        if not os.path.isfile(self.corpus_path) or not self.corpus_path.endswith('.csv'):
            raise ValueError(f"Corpus path '{self.corpus_path}' is not a .csv file for 'csv' file_type.")
        
        try:
            with open(self.corpus_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in tqdm(reader, desc="Loading .csv rows"):
                    if csv_id_col in row and csv_text_col in row:
                        doc_id = row[csv_id_col]
                        doc_text = row[csv_text_col]
                        self.documents[doc_id] = doc_text
                    else:
                        print(f"Warning: CSV row missing '{csv_id_col}' or '{csv_text_col}' columns: {row}")
        except Exception as e:
            print(f"Warning: Could not load CSV file {self.corpus_path}: {e}")

    def get_documents(self) -> Dict[Union[int, str], str]:
        """Returns the loaded documents."""
        return self.documents

    def get_document_by_id(self, doc_id: Union[int, str]) -> Union[str, None]:
        """Returns a single document by its ID."""
        return self.documents.get(doc_id)

# Example Usage
if __name__ == "__main__":
    print("--- Testing CorpusLoader ---")

    # --- Setup dummy corpus files/directory for testing ---
    test_corpus_dir = "test_corpus_txt"
    test_json_file = "test_corpus.json"
    test_csv_file = "test_corpus.csv"

    # Create dummy TXT files
    os.makedirs(test_corpus_dir, exist_ok=True)
    with open(os.path.join(test_corpus_dir, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("This is the first document about dogs and cats.")
    with open(os.path.join(test_corpus_dir, "doc2.txt"), "w", encoding="utf-8") as f:
        f.write("A second document talking about pets like cats and birds.")
    print(f"Created dummy .txt corpus in '{test_corpus_dir}'.")

    # Create dummy JSON file (list of objects)
    json_data_list = [
        {"id": "json_doc_A", "text": "JSON document A discusses software engineering."},
        {"id": "json_doc_B", "text": "Another JSON entry on machine learning algorithms."},
        {"id": "json_doc_C", "title": "Ignored Title", "text": "Data science is a fascinating field."} # Example with extra key
    ]
    with open(test_json_file, "w", encoding="utf-8") as f:
        json.dump(json_data_list, f, indent=4)
    print(f"Created dummy .json corpus file '{test_json_file}'.")

    # Create dummy CSV file
    csv_data = [
        {"doc_id": "csv_doc_X", "content": "CSV document X is about financial markets.", "author": "John Doe"},
        {"doc_id": "csv_doc_Y", "content": "Analyzing economic indicators from reports.", "author": "Jane Smith"},
        {"doc_id": "csv_doc_Z", "content": "Global trade and supply chains.", "author": "Alice Brown"}
    ]
    with open(test_csv_file, "w", encoding="utf-8", newline='') as f:
        fieldnames = ['doc_id', 'content', 'author']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Created dummy .csv corpus file '{test_csv_file}'.")

    # --- Test Loading ---

    # Test loading from TXT directory
    print("\n--- Loading from TXT directory ---")
    txt_loader = CorpusLoader(test_corpus_dir)
    txt_documents = txt_loader.load_documents(file_type='txt')
    print(f"Loaded TXT documents: {list(txt_documents.keys())}")
    print(f"Content of 'doc1': {txt_documents.get('doc1')[:50]}...")

    # Test loading from JSON file
    print("\n--- Loading from JSON file ---")
    json_loader = CorpusLoader(test_json_file)
    # Using 'id' and 'text' as default keys, so no need to pass them explicitly here
    json_documents = json_loader.load_documents(file_type='json')
    print(f"Loaded JSON documents: {list(json_documents.keys())}")
    print(f"Content of 'json_doc_A': {json_documents.get('json_doc_A')[:50]}...")

    # Test loading from CSV file with custom columns
    print("\n--- Loading from CSV file ---")
    csv_loader = CorpusLoader(test_csv_file)
    csv_documents = csv_loader.load_documents(file_type='csv', csv_id_col='doc_id', csv_text_col='content')
    print(f"Loaded CSV documents: {list(csv_documents.keys())}")
    print(f"Content of 'csv_doc_X': {csv_documents.get('csv_doc_X')[:50]}...")

    # Test loading with sequential integer IDs for TXT
    print("\n--- Loading from TXT directory with sequential IDs ---")
    txt_loader_int_id = CorpusLoader(test_corpus_dir)
    txt_documents_int_id = txt_loader_int_id.load_documents(file_type='txt', use_filename_as_id=False)
    print(f"Loaded TXT documents (sequential IDs): {list(txt_documents_int_id.keys())}")
    print(f"Content of Doc ID 0: {txt_documents_int_id.get(0)[:50]}...")


    # --- Cleanup dummy corpus files ---
    print("\nCleaning up dummy corpus files...")
    if os.path.exists(os.path.join(test_corpus_dir, "doc1.txt")):
        os.remove(os.path.join(test_corpus_dir, "doc1.txt"))
    if os.path.exists(os.path.join(test_corpus_dir, "doc2.txt")):
        os.remove(os.path.join(test_corpus_dir, "doc2.txt"))
    if os.path.exists(test_corpus_dir):
        os.rmdir(test_corpus_dir)
    
    if os.path.exists(test_json_file):
        os.remove(test_json_file)
    
    if os.path.exists(test_csv_file):
        os.remove(test_csv_file)
    print("Cleanup complete.")