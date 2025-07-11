# client_app.py

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
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the server at {url}. Please ensure the server is running.")
        return []
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")
        return []

def search_on_server(url: str, dataset: str, model_type: str, query: str, top_k: int) -> Optional[List[Dict]]:
    try:
        response = requests.post(f"{url}/search/{dataset}/{model_type}", json={"query": query, "top_k": top_k})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during search: {e}")
        try:
            # Attempt to show server's error details if available in response body
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

def evaluate_on_server(url: str, dataset: str, eval_k_values: List[int]) -> Optional[Dict]:
    try:
        response = requests.post(f"{url}/evaluate/{dataset}", json={"eval_k_values": eval_k_values})
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
        available_datasets,
        key="selected_dataset" # Streamlit handles this in session_state automatically
    )
else:
    st.warning("No datasets available or server not reachable. Please check server logs.")
    selected_dataset = None

st.sidebar.markdown("---")
st.sidebar.header("Actions")
selected_action = st.sidebar.radio(
    "Choose Action:",
    ["Search", "Query Optimization", "Run Evaluation"],
    key="selected_action" # Streamlit handles this in session_state automatically
)

# --- Main Content Area ---
if not selected_dataset:
    st.info("Please ensure the backend server is running and has loaded datasets.")
elif selected_action == "Search":
    st.header(f"Search in '{selected_dataset}' Dataset")

    # Use value from session_state for text_input
    query_input = st.text_input(
        "Enter your search query:", 
        value=st.session_state.search_query, 
        key="search_query_input_widget" # Unique key for the widget
    )
    
    col1, col2 = st.columns(2)
    with col1:
        # Use index to set initial value from session_state.selected_search_model
        model_choice = st.radio(
            "Select Search Model:",
            ('TF-IDF', 'BERT', 'Hybrid'),
            key="search_model_choice_widget", # Unique key for the widget
            horizontal=True,
            index=['TF-IDF', 'BERT', 'Hybrid'].index(st.session_state.selected_search_model)
        )
    with col2:
        # Use value from session_state for number_input
        top_k_input = st.number_input(
            "Number of top results:", 
            min_value=1, 
            value=st.session_state.top_k_search, 
            key="top_k_input_widget" # Unique key for the widget
        )

    # When search button is clicked, update session_state and perform search
    if st.button("🚀 Search", key="run_search_button"):
        if query_input:
            # Update session state with current widget values before search
            st.session_state.search_query = query_input
            st.session_state.selected_search_model = model_choice
            st.session_state.top_k_search = top_k_input
            
            with st.spinner(f"Searching with {model_choice} model..."):
                retrieved_results = search_on_server(
                    SERVER_URL, 
                    selected_dataset, 
                    model_choice.lower(), 
                    query_input, 
                    top_k_input
                )
                if retrieved_results:
                    st.session_state.search_results = retrieved_results # Store results
                else:
                    st.session_state.search_results = [] # Clear if no results or error
        else:
            st.warning("Please enter a query.")
            st.session_state.search_results = [] # Clear results if query is empty

    # Always display results if they exist in session_state
    if st.session_state.search_results:
        st.subheader(f"Results for '{st.session_state.search_query}' ({st.session_state.selected_search_model}):")
        for i, res in enumerate(st.session_state.search_results):
            st.markdown(f"**{i+1}. Document ID: `{res['doc_id']}` (Score: `{res['score']:.4f}`)**")
            with st.expander("Preview / Full Text"):
                st.write(res['text_preview'])
                # Ensure unique keys for buttons inside loops
                if st.button(f"Load Full Text for {res['doc_id']}", key=f"full_text_button_{res['doc_id']}"):
                    full_text = get_document_text_from_server(SERVER_URL, selected_dataset, res['doc_id'])
                    if full_text:
                        # Use unique key for the text_area too, so it doesn't disappear if another button is clicked
                        st.text_area(f"Full Text for {res['doc_id']}", full_text, height=300, key=f"full_text_display_{res['doc_id']}")
                    else:
                        st.warning("Could not retrieve full text.")
            st.markdown("---")
    elif st.session_state.search_query and not st.session_state.search_results and "run_search_button" in st.session_state and st.session_state["run_search_button"]:
        # Only show this if search was attempted and returned no results (avoids showing on initial load)
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
            st.session_state.original_query_optimize = original_query_optimize # Save original query
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
                    st.session_state.expanded_query = expanded_query # Store expanded query
                    st.success("Query Optimized!")
                    st.markdown(f"**Original Query:** `{original_query_optimize}`")
                    st.markdown(f"**Expanded Query:** `{expanded_query}`")
                else:
                    st.error("Failed to optimize query.")
                    st.session_state.expanded_query = ""
        else:
            st.warning("Please enter a query to optimize.")
            st.session_state.expanded_query = ""

    # Display expanded query and search option if expanded_query exists in session state
    if st.session_state.expanded_query:
        # Re-display expanded query even on rerun
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

        if st.button("🚀 Search with Expanded Query", key="run_expanded_search_button"):
            with st.spinner(f"Searching with {search_model_for_expanded} model and expanded query..."):
                expanded_results = search_on_server(
                    SERVER_URL, 
                    selected_dataset, 
                    search_model_for_expanded.lower(), 
                    st.session_state.expanded_query, # Use the stored expanded query
                    top_k_expanded
                )
                if expanded_results:
                    st.session_state.optimized_search_results = expanded_results # Store results
                else:
                    st.session_state.optimized_search_results = [] # Clear if no results
        
        # Display optimized search results
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

    # Store evaluation results in session state
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

    if st.button("📈 Run Evaluation", key="run_evaluation_button"):
        if selected_dataset and eval_k_values:
            with st.spinner("Running evaluation (this may take a while)..."):
                eval_results = evaluate_on_server(SERVER_URL, selected_dataset, eval_k_values)
                if eval_results:
                    st.session_state.evaluation_results = eval_results # Store results
                else:
                    st.session_state.evaluation_results = None # Clear on failure
        else:
            st.warning("Please select a dataset and enter valid k values.")
    
    # Always display evaluation results if they exist in session_state
    if st.session_state.evaluation_results:
        st.subheader(f"Evaluation Results for {selected_dataset}:")
        for model_name, metrics in st.session_state.evaluation_results.items():
            st.markdown(f"#### {model_name} Model")
            st.json(metrics) 
            st.markdown("---")
    elif "run_evaluation_button" in st.session_state and st.session_state["run_evaluation_button"] and not st.session_state.evaluation_results:
        st.info("Evaluation failed or no results returned.")