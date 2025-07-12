import streamlit as st
import requests
import json
from typing import Dict, List, Union, Optional

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000"

# --- Helper functions to interact with the server ---

@st.cache_data(ttl=3600) # Cache dataset list for an hour
def get_available_datasets_from_server(url: str) -> List[str]:
    try:
        response = requests.get(f"{url}/datasets")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # --- FIX STARTS HERE ---
        # The server is returning a list of strings directly, e.g., ["dataset1", "dataset2"]
        # So, we just return it as is.
        return response.json() 
        # --- FIX ENDS HERE ---

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the server at {url}. Please ensure the server is running.")
        return []
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")
        return []

def search_on_server(url: str, dataset: str, model_type: str, query: str, top_k: int, apply_spell_correction: bool = False) -> Optional[List[Dict]]:
    try:
        response = requests.post(
            f"{url}/search/{dataset}/{model_type}",
            json={
                "query": query,
                "top_k": top_k,
                "apply_spell_correction": apply_spell_correction # New parameter
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during search: {e}")
        try:
            st.json(response.json())
        except json.JSONDecodeError:
            st.text(f"Server returned non-JSON error: {response.text}")
        return None

# New function for clustered search
def search_with_clustering_on_server(url: str, dataset: str, query: str, top_k: int, target_cluster_id: Optional[int] = None, apply_spell_correction: bool = False) -> Optional[List[Dict]]:
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "apply_spell_correction": apply_spell_correction
        }
        if target_cluster_id is not None:
            payload["target_cluster_id"] = target_cluster_id

        response = requests.post(f"{url}/cluster/{dataset}/search", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during clustered search: {e}")
        try:
            st.json(response.json())
        except json.JSONDecodeError:
            st.text(f"Server returned non-JSON error: {response.text}")
        return None

def optimize_query_on_server(url: str, dataset: str, query: str, initial_search_model: str, top_n_docs: int, num_terms: int) -> Optional[str]:
    try:
        response = requests.post(
            f"{url}/optimize_query/{dataset}",
            json={
                "query": query,
                "initial_search_model": initial_search_model,
                "top_n_docs_for_prf": top_n_docs,
                "num_expansion_terms": num_terms
            }
        )
        response.raise_for_status()
        return response.json() # Expanded query is returned as a string
    except requests.exceptions.RequestException as e:
        st.error(f"Error during query optimization: {e}")
        try:
            st.json(response.json())
        except json.JSONDecodeError:
            st.text(f"Server returned non-JSON error: {response.text}")
        return None

# Updated evaluate_on_server to accept new parameters
def evaluate_on_server(url: str, dataset: str, eval_k_values: List[int],
                       use_clustering_for_bert_evaluation: bool,
                       use_prf_for_evaluation: bool,
                       prf_initial_model: Optional[str],
                       prf_top_n_docs: int,
                       prf_num_expansion_terms: int,
                       prf_final_model: Optional[str]) -> Optional[Dict]:
    try:
        payload = {
            "eval_k_values": eval_k_values,
            "use_clustering_for_bert_evaluation": use_clustering_for_bert_evaluation,
            "use_prf_for_evaluation": use_prf_for_evaluation,
            "prf_top_n_docs": prf_top_n_docs,
            "prf_num_expansion_terms": prf_num_expansion_terms
        }
        if prf_initial_model: # Only add if a model is selected
            payload["prf_initial_model"] = prf_initial_model
        if prf_final_model: # Only add if a final model is selected
            payload["prf_final_model"] = prf_final_model

        response = requests.post(f"{url}/evaluate/{dataset}", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during evaluation: {e}")
        try:
            st.json(response.json())
        except json.JSONDecodeError:
            st.text(f"Server returned non-JSON error: {response.text}")
        return None

def get_document_text_from_server(url: str, dataset: str, doc_id: Union[int, str]) -> Optional[str]:
    try:
        response = requests.get(f"{url}/document/{dataset}/{doc_id}")
        response.raise_for_status()
        return response.json().get('text')
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching document text: {e}")
        try:
            st.json(response.json())
        except json.JSONDecodeError:
            st.text(f"Server returned non-JSON error: {response.text}")
        return None

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="IR Search Engine Client")

st.title("📚 Information Retrieval Search Engine")
st.markdown("Interact with the IR backend server to perform searches, optimize queries, and run evaluations.")

# --- Initialize session state for persistent values ---
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_search_model' not in st.session_state:
    st.session_state.selected_search_model = 'TF-IDF'
if 'top_k_search' not in st.session_state:
    st.session_state.top_k_search = 10
if 'expanded_query' not in st.session_state:
    st.session_state.expanded_query = ""
if 'original_query_optimize' not in st.session_state:
    st.session_state.original_query_optimize = ""
if 'optimized_search_results' not in st.session_state:
    st.session_state.optimized_search_results = []
if 'search_type' not in st.session_state: # New: For clustered search option
    st.session_state.search_type = 'Standard Search'
if 'apply_spell_correction' not in st.session_state: # New: For spell correction
    st.session_state.apply_spell_correction = False


# --- Sidebar for Server/Dataset Selection ---
st.sidebar.header("Server Settings")
server_url_input = st.sidebar.text_input("FastAPI Server URL", SERVER_URL)
if server_url_input != SERVER_URL:
    SERVER_URL = server_url_input
    st.sidebar.info(f"Server URL updated to {SERVER_URL}. Please refresh the page if needed.")

available_datasets = get_available_datasets_from_server(SERVER_URL)

if available_datasets:
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset:",
        # --- FIX STARTS HERE ---
        # Now available_datasets is directly a list of strings
        available_datasets, 
        # --- FIX ENDS HERE ---
        key="selected_dataset"
    )
else:
    st.warning("No datasets available or server not reachable. Please check server logs.")
    selected_dataset = None

st.sidebar.markdown("---")
st.sidebar.header("Actions")
selected_action = st.sidebar.radio(
    "Choose Action:",
    ["Search", "Query Optimization", "Run Evaluation"],
    key="selected_action"
)

# --- Main Content Area ---
if not selected_dataset:
    st.info("Please ensure the backend server is running and has loaded datasets.")
elif selected_action == "Search":
    st.header(f"Search in '{selected_dataset}' Dataset")

    query_input = st.text_input(
        "Enter your search query:",
        value=st.session_state.search_query,
        key="search_query_input_widget"
    )

    col_search_options, col_k_sc = st.columns(2)
    with col_search_options:
        search_type = st.radio(
            "Select Search Type:",
            ('Standard Search', 'Clustered Search (BERT Only)'),
            key="search_type_widget",
            horizontal=True,
            index=['Standard Search', 'Clustered Search (BERT Only)'].index(st.session_state.search_type)
        )
        st.session_state.search_type = search_type # Update session state

    with col_k_sc:
        top_k_input = st.number_input(
            "Number of top results:",
            min_value=1,
            value=st.session_state.top_k_search,
            key="top_k_input_widget"
        )
        apply_spell_correction = st.checkbox(
            "Apply Spell Correction",
            value=st.session_state.apply_spell_correction,
            key="apply_spell_correction_checkbox"
        )
        st.session_state.apply_spell_correction = apply_spell_correction # Update session state

    target_cluster_id = None
    if search_type == 'Clustered Search (BERT Only)':
        st.info("Clustered search currently uses the BERT model internally.")
        target_cluster_id = st.number_input(
            "Target Cluster ID (Leave empty for auto-detection):",
            min_value=0,
            value=None, # Default to auto-detection
            key="target_cluster_id_input"
        )


    if st.button("🚀 Search", key="run_search_button"):
        if query_input:
            st.session_state.search_query = query_input
            st.session_state.top_k_search = top_k_input

            with st.spinner(f"Searching with {search_type}..."):
                retrieved_results = None
                if search_type == 'Standard Search':
                    model_choice = st.session_state.selected_search_model # Use the radio button state
                    retrieved_results = search_on_server(
                        SERVER_URL,
                        selected_dataset,
                        model_choice.lower(),
                        query_input,
                        top_k_input,
                        apply_spell_correction # Pass spell correction option
                    )
                elif search_type == 'Clustered Search (BERT Only)':
                    retrieved_results = search_with_clustering_on_server(
                        SERVER_URL,
                        selected_dataset,
                        query_input,
                        top_k_input,
                        target_cluster_id,
                        apply_spell_correction # Pass spell correction option
                    )

                if retrieved_results:
                    st.session_state.search_results = retrieved_results
                else:
                    st.session_state.search_results = []
        else:
            st.warning("Please enter a query.")
            st.session_state.search_results = []

    # Display search model choice for Standard Search only
    if search_type == 'Standard Search':
        st.session_state.selected_search_model = st.radio(
            "Select Standard Search Model:",
            ('TF-IDF', 'BERT', 'Hybrid'),
            key="standard_search_model_choice_widget",
            horizontal=True,
            index=['TF-IDF', 'BERT', 'Hybrid'].index(st.session_state.selected_search_model)
        )


    if st.session_state.search_results:
        st.subheader(f"Results for '{st.session_state.search_query}' ({search_type}):")
        for i, res in enumerate(st.session_state.search_results):
            st.markdown(f"**{i+1}. Document ID: `{res['doc_id']}` (Score: `{res['score']:.4f}`)**")
            with st.expander("Preview / Full Text"):
                st.write(res['text_preview'])
                if st.button(f"Load Full Text for {res['doc_id']}", key=f"full_text_button_{res['doc_id']}"):
                    full_text = get_document_text_from_server(SERVER_URL, selected_dataset, res['doc_id'])
                    if full_text:
                        st.text_area(f"Full Text for {res['doc_id']}", full_text, height=300, key=f"full_text_display_{res['doc_id']}")
                    else:
                        st.warning("Could not retrieve full text.")
            st.markdown("---")
    elif query_input and not st.session_state.search_results and st.button("🚀 Search", key="run_search_button_recheck", help="Hidden re-check button"): # Add a unique key here too
        st.info("No results found for your query.")


elif selected_action == "Query Optimization":
    st.header(f"Query Optimization (PRF) for '{selected_dataset}'")

    original_query_optimize = st.text_input(
        "Enter query to optimize:",
        value=st.session_state.original_query_optimize,
        key="optimize_query_input_widget"
    )

    st.subheader("PRF Parameters")
    col1_prf, col2_prf, col3_prf = st.columns(3)
    with col1_prf:
        prf_initial_model = st.radio(
            "Initial Search Model:",
            ('TF-IDF', 'BERT'),
            key="prf_model_choice"
        )
    with col2_prf:
        prf_top_n_docs = st.number_input(
            "Top N Docs for Feedback:", min_value=1, value=5, key="prf_top_docs"
        )
    with col3_prf:
        prf_num_terms = st.number_input(
            "Num Terms to Add:", min_value=1, value=3, key="prf_num_terms"
        )

    if st.button("✨ Optimize Query", key="run_optimize_button"):
        if original_query_optimize:
            st.session_state.original_query_optimize = original_query_optimize
            with st.spinner("Optimizing query..."):
                expanded_query = optimize_query_on_server(
                    SERVER_URL,
                    selected_dataset,
                    original_query_optimize,
                    prf_initial_model,
                    prf_top_n_docs,
                    prf_num_terms
                )
                if expanded_query:
                    st.session_state.expanded_query = expanded_query
                    st.success("Query Optimized!")
                    st.markdown(f"**Original Query:** `{original_query_optimize}`")
                    st.markdown(f"**Expanded Query:** `{expanded_query}`")
                else:
                    st.error("Failed to optimize query.")
                    st.session_state.expanded_query = ""
        else:
            st.warning("Please enter a query to optimize.")
            st.session_state.expanded_query = ""

    if st.session_state.expanded_query:
        st.markdown(f"**Original Query:** `{st.session_state.original_query_optimize}`")
        st.markdown(f"**Expanded Query:** `{st.session_state.expanded_query}`")

        st.subheader("Perform Search with Expanded Query?")
        search_model_for_expanded = st.radio(
            "Select Search Model for Expanded Query:",
            ('TF-IDF', 'BERT', 'Hybrid'),
            key="expanded_search_model",
            horizontal=True
        )
        top_k_expanded = st.number_input("Number of top results for expanded search:", min_value=1, value=10, key="expanded_top_k")

        # Adding spell correction option for the expanded query search as well
        apply_spell_correction_expanded = st.checkbox(
            "Apply Spell Correction to Expanded Search",
            value=st.session_state.apply_spell_correction, # Default to main search setting
            key="apply_spell_correction_expanded_checkbox"
        )

        if st.button("🚀 Search with Expanded Query", key="run_expanded_search_button"):
            with st.spinner(f"Searching with {search_model_for_expanded} model and expanded query..."):
                expanded_results = search_on_server( # Use search_on_server for consistency
                    SERVER_URL,
                    selected_dataset,
                    search_model_for_expanded.lower(),
                    st.session_state.expanded_query,
                    top_k_expanded,
                    apply_spell_correction_expanded # Pass spell correction
                )
                if expanded_results:
                    st.session_state.optimized_search_results = expanded_results
                else:
                    st.session_state.optimized_search_results = []

        if st.session_state.optimized_search_results:
            st.subheader(f"Results for Expanded Query ({search_model_for_expanded}):")
            for i, res in enumerate(st.session_state.optimized_search_results):
                st.markdown(f"**{i+1}. Document ID: `{res['doc_id']}` (Score: `{res['score']:.4f}`)**")
                with st.expander("Preview / Full Text"):
                    st.write(res['text_preview'])
                    if st.button(f"Load Full Text for {res['doc_id']} (Expanded)", key=f"full_text_exp_button_{res['doc_id']}"):
                        full_text = get_document_text_from_server(SERVER_URL, selected_dataset, res['doc_id'])
                        if full_text:
                            st.text_area(f"Full Text for {res['doc_id']} (Expanded)", full_text, height=300, key=f"full_text_exp_display_{res['doc_id']}")
                        else:
                            st.warning("Could not retrieve full text.")
                st.markdown("---")
        elif "run_expanded_search_button" in st.session_state and st.session_state["run_expanded_search_button"]:
             st.info("No results found for the expanded query.")


elif selected_action == "Run Evaluation":
    st.header(f"Run Evaluation on '{selected_dataset}' Dataset")

    eval_k_values_str = st.text_input("Enter k values for evaluation (comma-separated):", "1,5,10,20", key="eval_k_values_input")
    eval_k_values = []
    try:
        eval_k_values = [int(k.strip()) for k in eval_k_values_str.split(',') if k.strip()]
        if not eval_k_values:
            st.warning("Please enter valid k values.")
    except ValueError:
        st.error("Invalid k values. Please enter comma-separated integers.")
        eval_k_values = []

    st.subheader("Evaluation Options")
    col_eval_opt1, col_eval_opt2 = st.columns(2)

    with col_eval_opt1:
        use_clustering_for_bert_evaluation = st.checkbox(
            "Include 'BERT + Clustering' in Evaluation",
            key="eval_clustering_checkbox"
        )
    with col_eval_opt2:
        use_prf_for_evaluation = st.checkbox(
            "Include 'Query Optimization (PRF)' in Evaluation",
            key="eval_prf_checkbox"
        )

    # PRF Options (conditionally displayed)
    if use_prf_for_evaluation:
        st.markdown("---")
        st.subheader("Query Optimization (PRF) Evaluation Parameters")
        col_prf_eval1, col_prf_eval2, col_prf_eval3, col_prf_eval4 = st.columns(4)
        with col_prf_eval1:
            prf_initial_model = st.selectbox(
                "PRF Initial Search Model:",
                ('TF-IDF', 'BERT'),
                key="eval_prf_initial_model"
            )
        with col_prf_eval2:
            prf_top_n_docs = st.number_input(
                "PRF Top N Docs for Feedback:", min_value=1, value=5, key="eval_prf_top_docs"
            )
        with col_prf_eval3:
            prf_num_expansion_terms = st.number_input(
                "PRF Num Terms to Add:", min_value=1, value=3, key="eval_prf_num_terms"
            )
        with col_prf_eval4:
            prf_final_model = st.selectbox(
                "PRF Final Search Model:",
                ('TF-IDF', 'BERT', 'Hybrid'),
                index=2, # Default to Hybrid
                help="Model to use for the final search with the expanded query.",
                key="eval_prf_final_model"
            )
    else:
        # Set PRF parameters to default/None if not used, to ensure clean payload
        prf_initial_model = None
        prf_top_n_docs = 0
        prf_num_expansion_terms = 0
        prf_final_model = None


    # Store evaluation results in session state
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

    if st.button("📈 Run Evaluation", key="run_evaluation_button"):
        if selected_dataset and eval_k_values:
            # Validate PRF initial model if PRF is enabled
            if use_prf_for_evaluation and not prf_initial_model:
                st.error("Please select an 'Initial Search Model' for PRF evaluation.")
            else:
                with st.spinner("Running evaluation (this may take a while)..."):
                    eval_results = evaluate_on_server(
                        SERVER_URL,
                        selected_dataset,
                        eval_k_values,
                        use_clustering_for_bert_evaluation,
                        use_prf_for_evaluation,
                        prf_initial_model,
                        prf_top_n_docs,
                        prf_num_expansion_terms,
                        prf_final_model
                    )
                    if eval_results:
                        st.session_state.evaluation_results = eval_results
                    else:
                        st.session_state.evaluation_results = None
        else:
            st.warning("Please select a dataset and enter valid k values.")

    # Always display evaluation results if they exist in session_state
    if st.session_state.evaluation_results:
        st.subheader(f"Evaluation Results for {selected_dataset}:")
        # Sort results by model name for consistent display
        sorted_model_names = sorted(st.session_state.evaluation_results.keys())
        for model_name in sorted_model_names:
            metrics = st.session_state.evaluation_results[model_name]
            st.markdown(f"#### {model_name} Model")
            st.json(metrics)
            st.markdown("---")
    elif "run_evaluation_button" in st.session_state and st.session_state["run_evaluation_button"] and not st.session_state.evaluation_results:
        st.info("Evaluation failed or no results returned.")