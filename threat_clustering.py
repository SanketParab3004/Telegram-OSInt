import os
import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CLUSTER_FILENAME = "threat_clusters.csv"

def handle_missing_values(df):
    """
    Handle missing values in the processed_messages column.
    """
    logging.info("Handling missing values in 'processed_messages' column...")
    df['processed_messages'] = df['processed_messages'].fillna('')  # Replace NaN with an empty string
    return df

def perform_clustering(df):
    """
    Perform clustering using KMeans and return the dataframe with cluster labels.
    """
    logging.info("Performing clustering on threat data...")

    # Preprocess text data using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(df['processed_messages'])

    # Reduce dimensionality using TruncatedSVD (optional, improves performance)
    svd = TruncatedSVD(n_components=100)  # Reduce dimensions to 100 components
    X_reduced = svd.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)  # Explicitly setting n_init
    df.loc[:, 'Cluster_Label'] = kmeans.fit_predict(X_reduced)  # Use .loc to avoid the warning

    # Evaluate clustering performance
    silhouette_avg = silhouette_score(X_reduced, df['Cluster_Label'])
    logging.info(f"Silhouette Score for Clustering: {silhouette_avg:.4f}")

    return df

def save_clustering_results(df, input_folder):
    """
    Save the clustering results to a CSV file in the same folder as the input file.
    """
    output_file_path = os.path.join(input_folder, CLUSTER_FILENAME)
    df.to_csv(output_file_path, index=False)
    logging.info(f"Clustering results saved to {output_file_path}")

def main():
    try:
        # Locate the latest prediction data folder
        prediction_folder = "Threats"
        latest_folder = max(
            [os.path.join(prediction_folder, folder) for folder in os.listdir(prediction_folder)],
            key=os.path.getmtime
        )
        
        predictions_file = os.path.join(latest_folder, "threat_predictions.csv")
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Prediction file not found: {predictions_file}")

        # Load predictions data
        df = pd.read_csv(predictions_file)
        logging.info(f"Loaded predictions data from {predictions_file}.")

        # Handle missing values in processed_messages
        df = handle_missing_values(df)

        # Perform clustering on the threat data
        df_with_clusters = perform_clustering(df)

        # Save clustering results in the same folder as the input file
        save_clustering_results(df_with_clusters, latest_folder)

    except Exception as e:
        logging.error(f"An error occurred while clustering the data: {e}")

if __name__ == "__main__":
    main()
