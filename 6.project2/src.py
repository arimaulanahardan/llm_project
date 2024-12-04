import pandas as pd 
import json
from sklearn.cluster import KMeans
from langchain_ollama import ChatOllama

llm = ChatOllama(name="chat_llama3", model="llama3.2:1b", temperature=0)

# function to read file and metadata
def read_data(filepath):
    df = pd.read_csv(filepath)
    return df

# function to read metadata
def read_json(filepath):
    """
    Reads a JSON file from the specified file path.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data as a Python dictionary or list.
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def preprocess_for_clustering(df, selected_features):
    """
    Preprocesses the DataFrame for clustering analysis.

    Args:
        df (pd.DataFrame): The input DataFrame.
        selected_features (list): List of selected column names.

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for clustering.
    """
    df2 = df[selected_features].copy()
    categorical_features = df2.select_dtypes(include='object').columns
    numerical_features = df2.select_dtypes(exclude='object').columns

    # Step 1: Handle missing values
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Filling missing values with mean for numerical and mode for categorical.")
        for col in numerical_features:
            df2[col] = df2[col].fillna(df[col].mean())
        for col in categorical_features:
            df2[col] = df2[col].fillna(df[col].mode()[0])

    # Step 2: Encode categorical features using pd.get_dummies
    df_encoded = pd.get_dummies(df2, columns=categorical_features, drop_first=True)

    return df_encoded

def filter_metadata(metadata,selected_faetures):
    metadata_select = []
    for item in metadata :
        if item['feature_name'] in selected_faetures:
            metadata_select.append(item)
    return metadata_select


def cluster_data(df, n_clusters):
    """
    Clusters the input data using KMeans and returns the labeled DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be clustered.
        n_clusters (int): The number of clusters to form.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Cluster' column containing cluster labels.
    """
    try:
        # Initialize KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Fit the model and predict cluster labels
        cluster_labels = kmeans.fit_predict(df)
        
        # Add cluster labels to the DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels

        df_agg = df_with_clusters.groupby('Cluster').mean()
        
        return df_agg
    except Exception as e:
        raise ValueError(f"Error during clustering: {e}")

def llm_model(metadata,n_clusters,df_agg):
    cluster_result = df_agg.reset_index().to_dict('records')
    prompt = """
    You are AI assistant to analyze cluster result, you will help to create a cluster definition based on all the input you got. 
    You will get the cluster result explaining the distribution of the cluster, and based on this you need to make the label or definition of the cluster.
    Here is the input description :

    1. metadata : the description about the data from user
    2. n_cluster : number of cluster
    2. cluster_result : Aggregated data for each result, explaining the distribution of the data

    output will be :
    1. cluster : based on input
    2. cluster_name : name of the cluster based on the persona you think of
    3. definition : the reason you give the cluster name for that cluster

    Follow this procedure when creating the result :
    1. Follow the output template, put the output as json format
    ```json
    ['cluster':1,
    'cluster_name:cluster name,
    'defintion':Definition of cluster]
    ````
    2. Define number of cluster based on the input from user 
    3. Write everything after output

    Input :
    metadata = {metadata}
    n_cluster = {n_cluster}
    cluster_result = {cluster_result}

    Output :
    """

    llm_prompt = prompt.format(metadata=metadata, n_cluster=n_clusters, cluster_result = cluster_result)

    result = llm.invoke(llm_prompt)
    return result.content

def parse_output(free_text):
    """
    Parse free text containing a JSON structure with additional explanations and extract cluster information.
    
    Parameters:
        free_text (str): Input string containing the JSON structure and explanation.
    
    Returns:
        list[dict]: Parsed list of dictionaries with cluster details.
    """
    # Extract the JSON part from the text
    json_start = free_text.find("[")
    json_end = free_text.rfind("]") + 1  # Include the closing bracket
    
    if json_start == -1 or json_end == -1:
        raise ValueError("No valid JSON structure found in the input text.")
    
    json_text = free_text[json_start:json_end]
    
    # Parse the JSON
    try:
        clusters = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError("Failed to decode JSON from the input text.") from e
    
    return clusters

