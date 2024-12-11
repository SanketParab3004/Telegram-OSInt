import os
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def locate_merged_data_file():
    base_folder = "processed_data"
    try:
        # Find all subfolders in the base folder
        subfolders = [
            os.path.join(base_folder, folder)
            for folder in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, folder))
        ]
        if not subfolders:
            logging.error("No processed data folders found.")
            return None

        # Find the most recent subfolder
        latest_folder = max(subfolders, key=os.path.getmtime)

        # Construct the path to `merged_data_for_svm.csv`
        file_path = os.path.join(latest_folder, "merged_data_for_svm.csv")
        logging.info(f"Looking for merged data file at: {file_path}")

        # Check if the file exists
        if os.path.exists(file_path):
            logging.info(f"Merged data file found: {file_path}")
            return file_path
        else:
            logging.error("Merged data file not found in the latest folder.")
            return None
    except Exception as e:
        logging.error(f"Error locating merged data file: {e}")
        return None


def load_csv(file_path, sample_size=None):
    """
    Load CSV data, optionally sampling a smaller subset for testing.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}, total rows: {len(data)}")

        # If sample_size is specified, take a random sample
        if sample_size:
            data = data.sample(n=sample_size, random_state=42)
            logging.info(f"Sampled {sample_size} rows for testing.")

        logging.info(f"Sampled data preview:\n{data.head()}")
        return data
    except Exception as e:
        logging.error(f"Error loading CSV from {file_path}: {e}")
        return pd.DataFrame()

def create_feature_matrix(df):
    """
    Create a feature matrix from the preprocessed DataFrame.
    """
    if df.empty:
        logging.error("No data to process.")
        return None, None

    # Drop rows with NaN values in the 'processed_messages' column
    df = df.dropna(subset=['processed_messages'])

    # Convert text data into numerical format using TF-IDF Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(df['processed_messages'])

    logging.info(f"Feature matrix created with shape: {X.shape}")
    return X, df  # Return both the feature matrix and the DataFrame with metadata

def apply_one_class_svm(X, df):
    """
    Apply One-Class SVM for anomaly detection.
    """
    logging.info("Applying anomaly detection using One-Class SVM...")

    # Standardize the feature matrix
    scaler = StandardScaler(with_mean=False)  # with_mean=False to keep sparse matrix format
    X_scaled = scaler.fit_transform(X)

    # Initialize One-Class SVM model
    one_class_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)

    try:
        # Fit the model
        one_class_svm.fit(X_scaled)

        # Predict anomalies (-1 for anomalies, 1 for normal data)
        anomaly_labels = one_class_svm.predict(X_scaled)

        # Count and log the number of anomalies detected
        num_anomalies = (anomaly_labels == -1).sum()
        logging.info(f"Number of anomalies detected: {num_anomalies}")

        # Identify the anomalies and save them to a new DataFrame
        anomalies_df = df[anomaly_labels == -1].copy()
        anomalies_df['Anomaly_Label'] = 'Anomaly'  # Label anomalies for clarity

        # Create a timestamped folder under the Anomaly directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join("Anomaly", timestamp)
        os.makedirs(folder_path, exist_ok=True)

        # Save the anomalies to a CSV file inside the timestamped folder
        output_file = os.path.join(folder_path, "detected_anomalies.csv")
        anomalies_df.to_csv(output_file, index=False)
        logging.info(f"Anomalies saved to {output_file}")

        # Plot the results
        plt.figure(figsize=(10, 6))
        normal_indices = np.where(anomaly_labels == 1)[0]  # Indices of normal points
        anomaly_indices = np.where(anomaly_labels == -1)[0]  # Indices of anomalies

        # Plot normal points
        plt.scatter(normal_indices, [1] * len(normal_indices), c='blue', label='Normal (Blue Dots)', alpha=0.7)

        # Plot anomaly points
        plt.scatter(anomaly_indices, [-1] * len(anomaly_indices), c='red', label='Anomaly (Red Dots)', alpha=0.7)

        plt.xlabel('Data Point Index')
        plt.ylabel('Anomaly Label')
        plt.title('One-Class SVM Anomaly Detection Results')
        plt.legend(loc='best')  # Add legend for clarity
        save_plot(plt, "one_class_svm_anomaly_detection", folder_path)

        # Bar chart to show counts of normal and anomalous points
        normal_count = (anomaly_labels == 1).sum()
        anomaly_count = (anomaly_labels == -1).sum()

        plt.figure(figsize=(8, 6))
        plt.bar(['Normal', 'Anomalous'], [normal_count, anomaly_count], color=['blue', 'red'])
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.title('Summary of Anomaly Detection Results')
        plt.savefig(os.path.join(folder_path, "anomaly_detection_summary.png"))
        logging.info(f"Anomaly detection summary plot saved to {folder_path}")
        plt.show()

    except Exception as e:
        logging.error(f"Error fitting One-Class SVM: {e}")
        raise

def save_plot(fig, plot_name, folder_path):
    """
    Save plot to the specified folder.
    """
    file_path = os.path.join(folder_path, f"{plot_name}.png")
    fig.savefig(file_path)
    logging.info(f"Figure saved to {file_path}")

def main(sample_size=None):
    """
    Main function to execute the entire pipeline.
    """
    file_path = locate_merged_data_file()

    if not file_path:
        return

    # Load the merged data with optional sampling
    merged_data = load_csv(file_path, sample_size=sample_size)

    # Create the feature matrix from the merged data
    X, df = create_feature_matrix(merged_data)
    if X is None:
        logging.error("Feature matrix creation failed. Exiting.")
        return

    # Apply One-Class SVM for anomaly detection
    apply_one_class_svm(X, df)

if __name__ == "__main__":
    # Specify sample size for testing (e.g., 100 rows)
    main(sample_size=100)
