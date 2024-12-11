import os
import pandas as pd
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_PROCESSED_DATA_DIR = "processed_data"  # Base directory for processed data
MODEL_PATH = "cyber_threat_detection_model.pkl"  # Path to the pre-trained model
BASE_OUTPUT_DIR = "Threats"  # Base directory for storing threat prediction results

# Create timestamped folder for storing results
def create_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(BASE_OUTPUT_DIR, timestamp)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Function to locate the latest processed folder based on modification time
def locate_latest_processed_folder(base_path=BASE_PROCESSED_DATA_DIR):
    logging.info("Finding the latest processed data folder...")
    subfolders = [
        os.path.join(base_path, folder)
        for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder)) and folder.startswith("processed_data_")
    ]
    
    if not subfolders:
        raise FileNotFoundError(f"No processed data folders found in {base_path}")
    
    latest_folder = max(subfolders, key=os.path.getmtime)
    logging.info(f"Latest processed folder: {latest_folder}")
    return latest_folder

# Function to load the pre-trained model
def load_model(model_path):
    logging.info("Loading the threat detection model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    logging.info("Model loaded successfully.")
    return model

# Function to preprocess the data and create labels if they don't exist
def preprocess_and_create_labels(df):
    logging.info("Preprocessing data and creating labels...")
    
    # Check if 'label' column exists, if not, create it
    if 'label' not in df.columns:
        logging.info("Label column not found. Creating 'label' column...")
        
        # Define rules for labeling
        def label_data(row):
            # Rule 1: If ioc_value contains suspicious keywords
            suspicious_keywords = ['malware', 'phishing', 'hack', 'attack', 'leak']
            if any(keyword in str(row['ioc_value']).lower() for keyword in suspicious_keywords):
                return 1  # threat
            
            # Rule 2: If processed_messages contain threat-related keywords
            threat_keywords = ['leak', 'breach', 'hack', 'attack']
            if any(keyword in str(row['processed_messages']).lower() for keyword in threat_keywords):
                return 1  # threat
            
            # Rule 3: If the participant is unverified or offline
            if row['is_verified'] == False or 'offline' in str(row['status']).lower():
                return 1  # threat
            
            # Rule 4: If the ioc_type is suspicious (e.g., url or hash known to be dangerous)
            suspicious_ioc_types = ['url', 'hash']
            if row['ioc_type'] in suspicious_ioc_types:
                return 1  # threat

            # If none of the conditions apply, label as non-threat
            return 0  # non-threat
        
        # Apply labeling function to each row
        df['label'] = df.apply(label_data, axis=1)
        logging.info("Labels created successfully.")
    
    return df

# Function to preprocess the data and convert text to numeric features
def preprocess_data(df):
    """
    Preprocess data by cleaning and converting non-numeric features to numeric.
    """
    logging.info("Preprocessing data...")

    # Ensure that required columns exist
    required_columns = [
        "processed_messages", "ioc_type", "ioc_value", "original_message",
        "participant_name", "status", "is_verified", "entity", "type", "value"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            return None, None, None

    # Handle NaN in 'processed_messages' column by replacing them with an empty string
    df['processed_messages'] = df['processed_messages'].fillna('')

    # Drop rows with missing target labels
    df = df.dropna(subset=["label"])

    # Remove any rows with non-numeric features or convert them to numeric
    numeric_df = df.select_dtypes(include=[float, int])

    # If there are text columns, convert them to numeric using TF-IDF
    if 'processed_messages' in df.columns:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        X_text = vectorizer.fit_transform(df['processed_messages'])
        numeric_df = pd.concat([numeric_df, pd.DataFrame(X_text.toarray())], axis=1)

    # Return the numeric features and the labels
    X = numeric_df.drop(columns=["label"])
    y = df["label"]
    return X, y, vectorizer


# Function to make predictions
def predict_threats(data_path, model, output_path):
    logging.info(f"Loading dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Preprocess data and create labels if necessary
    df = preprocess_and_create_labels(df)
    
    # Preprocess data for model input
    X, y, vectorizer = preprocess_data(df)
    
    # Split into training and testing datasets (optional, can be omitted if using pre-trained model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model (if not using pre-trained)
    logging.info("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate model performance on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    # Make predictions for the entire dataset
    logging.info("Making predictions...")
    df['Threat_Prediction'] = model.predict(X)
    
    # Save predictions to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    df.to_csv(output_path, index=False)
    logging.info(f"Threat predictions saved to {output_path}")

# Main function
def main():
    try:
        # Locate the latest processed data folder
        latest_folder = locate_latest_processed_folder()
        
        # Construct the path to the dataset
        data_path = os.path.join(latest_folder, "merged_data_for_svm.csv")
        logging.info(f"Using dataset: {data_path}")
        
        # Load the model
        model = load_model(MODEL_PATH)
        
        # Create a folder for this prediction run
        output_folder = create_output_folder()
        logging.info(f"Prediction results will be saved to: {output_folder}")

        # Perform predictions and save the results
        output_path = os.path.join(output_folder, "threat_predictions.csv")
        predict_threats(data_path, model, output_path)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
