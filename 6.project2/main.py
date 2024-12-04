import pandas as pd
import streamlit as st
import os

from src import read_data, read_json, preprocess_for_clustering, cluster_data, llm_model,filter_metadata, parse_output

# Initialize session state for parameters
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "metadata_file" not in st.session_state:
    st.session_state["metadata_file"] = None
if "feature_columns" not in st.session_state:
    st.session_state["feature_columns"] = []
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = []
if "n_clusters" not in st.session_state:
    st.session_state["n_clusters"] = 3
if "data_filepath" not in st.session_state:
    st.session_state["data_filepath"] = None
if "metadata_filepath" not in st.session_state:
    st.session_state["metadata_filepath"] = None
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

# Streamlit App
st.title("Clustering Tool")

# Sidebar for uploading files
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV)", type=["csv"])
metadata_file = st.sidebar.file_uploader("Upload your metadata file (JSON)", type=["json"])

if st.sidebar.button("Upload File"):
    if uploaded_file:
        # Save uploaded file in session state
        st.session_state["uploaded_file"] = uploaded_file

        # Save the file to the designated folder
        st.session_state["data_filepath"] = os.path.join("data/upload", uploaded_file.name)
        with open(st.session_state["data_filepath"], "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read data and update session state
        st.session_state["data"]  = read_data(st.session_state["data_filepath"])
        st.session_state["feature_columns"] = st.session_state["data"] .columns.tolist()
    else:
        st.write("Please upload a CSV file to display its features.")

    if metadata_file:
        # Save uploaded file in session state
        st.session_state["metadata_file"] = metadata_file

        # Save the file to the designated folder
        st.session_state["metadata_filepath"] = os.path.join("data/upload", metadata_file.name)
        with open(st.session_state["metadata_filepath"], "wb") as f:
            f.write(metadata_file.getbuffer())

        st.session_state["metadata"] = read_json(st.session_state["metadata_filepath"])

if len(st.session_state["data"]) > 0 :
    st.dataframe(st.session_state["data"].head(20))

# Sidebar dropdown for feature selection
st.sidebar.header("Select Features for Clustering")
selected_features = st.sidebar.multiselect(
    "Choose features from the dataset:",
    options=st.session_state["feature_columns"],
    default=st.session_state["feature_columns"][:2]
    if st.session_state["feature_columns"]
    else [],
)

# Update session state with selected features
st.session_state["selected_features"] = selected_features

# Sidebar slider for the number of clusters
n_clusters = st.sidebar.slider(
    "Number of Clusters", 
    min_value=1, 
    max_value=10, 
    value=st.session_state["n_clusters"]
)

# Update session state with the number of clusters
st.session_state["n_clusters"] = n_clusters

# Button to run clustering analysis
if st.sidebar.button("Run Clustering Analysis"):

    if st.session_state["uploaded_file"] and st.session_state["selected_features"]:

        # Preprocess data for clustering
        processed_data = preprocess_for_clustering(st.session_state["data"], st.session_state["selected_features"])
        filtere_metadata = filter_metadata(st.session_state["metadata"],st.session_state["selected_features"])

        # Perform clustering
        df_clusters = cluster_data(processed_data, st.session_state["n_clusters"])
        
        # LLM call
        llm_result = llm_model(st.session_state["data"],st.session_state["n_clusters"],df_clusters)
        cluster_result = parse_output(llm_result)

        # Display the results
        st.write("Clustering Results:")
        st.dataframe(df_clusters)
        st.write(cluster_result)
    else:
        st.warning("Please upload a file and select features before running the clustering analysis.")
