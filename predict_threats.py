import os
import pickle
import pandas as pd
from data_preprocessing import preprocess_data  # Import the preprocessing script

def locate_latest_processed_data():
    """
    Locates the most recently created dynamic folder inside the 'processed_data' directory.

    Returns:
        str: Path to the latest dynamic folder or None if no folder is found.
    """
    processed_data_base = "processed_data"
    if not os.path.exists(processed_data_base):
        print(f"Processed data folder '{processed_data_base}' does not exist.")
        return None

    subfolders = [
        os.path.join(processed_data_base, folder)
        for folder in os.listdir(processed_data_base)
        if os.path.isdir(os.path.join(processed_data_base, folder))
    ]

    if not subfolders:
        print("No subfolders found in the 'processed_data' directory.")
        return None

    latest_folder = max(subfolders, key=os.path.getmtime)
    print(f"Located latest processed data folder: {latest_folder}")
    return latest_folder


def load_csv_files(processed_data_folder):
    """
    Loads the CSV files from the given processed data folder.

    Args:
        processed_data_folder (str): Path to the folder containing processed CSV files.

    Returns:
        dict: A dictionary with DataFrame names as keys and loaded DataFrames as values.
    """
    file_mapping = {
        "messages": "processed_messages_combined.csv",
        "iocs": "processed_iocs_combined.csv",
        "all_entities": "processed_all_entities_combined.csv",
        "participants": "processed_participants_combined.csv",
    }

    dataframes = {}
    for key, filename in file_mapping.items():
        file_path = os.path.join(processed_data_folder, filename)
        if os.path.exists(file_path):
            dataframes[key] = pd.read_csv(file_path)
            print(f"Loaded '{key}' data from {file_path} with {len(dataframes[key])} records.")
        else:
            print(f"File '{filename}' not found in {processed_data_folder}. Skipping.")

    return dataframes


def predict_new_data():
    """
    Load preprocessed data, apply the trained model, and output predictions.
    """
    try:
        # Load the trained model
        with open('cyber_threat_detection_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")

        # Load the vectorizer used during training
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Vectorizer loaded successfully.")

        # Preprocess new data
        X_new, _ = preprocess_data()
        if X_new is None or X_new.empty:
            print("No data available for prediction.")
            return

        # Apply vectorizer to preprocess the input (ensure X_new is transformed using the same vectorizer)
        X_new_tfidf = vectorizer.transform(X_new)

        # Ensure the input matches model expectations (e.g., feature columns)
        predictions = model.predict(X_new_tfidf)
        print("Predictions:", predictions)

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == '__main__':
    predict_new_data()
