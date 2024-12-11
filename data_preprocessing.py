import os
import json
import pandas as pd
import logging
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Locate JSON files with dynamic names in the "output" directory
def locate_files():
    base_path = "output"
    paths = {
        "messages": [],
        "iocs": [],
        "participants": [],
        "all_entities": [],
        "direct_messages": []
    }

    try:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith("messages_") and file.endswith(".json"):
                    paths["messages"].append(os.path.join(root, file))
                elif file.startswith("iocs_") and file.endswith(".json"):
                    paths["iocs"].append(os.path.join(root, file))
                elif file.startswith("participants_") and file.endswith(".json"):
                    paths["participants"].append(os.path.join(root, file))
                elif file == "all_entities.json":
                    paths["all_entities"].append(os.path.join(root, file))
                elif 'direct_message_' in file:
                    paths["direct_messages"].append(root)  # Save folder name instead of file
        logging.info(f"Paths found: {paths}")
    except Exception as e:
        logging.error(f"Error in locating files: {e}")
    return paths

# Load JSON data
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {file_path}: {data[:5]}")  # Debug: print first 5 entries to see the structure
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return []

# Process messages data
def preprocess_messages(messages):
    processed = []
    try:
        for message in messages:
            text = message.get('message', '')
            text = text.strip()  # Clean up the text if needed
            processed.append(text)
    except Exception as e:
        logging.error(f"Error processing messages: {e}")
    return processed

# Process IOCs data
def extract_iocs(iocs):
    ioc_data = []
    try:
        for ioc in iocs:
            ioc_type = ioc.get('ioc_type', '')
            ioc_value = ioc.get('ioc_value', '')
            original_message = ioc.get('original_message', '')
            ioc_data.append({
                'ioc_type': ioc_type,
                'ioc_value': ioc_value,
                'original_message': original_message
            })
    except Exception as e:
        logging.error(f"Error extracting IOCs: {e}")
    return ioc_data

# Process participants data
def extract_participant_features(participants):
    features = []
    try:
        for participant in participants:
            first_name = participant.get('first_name', '')
            last_name = participant.get('last_name', '')
            full_name = f"{first_name} {last_name}".strip()
            
            status = participant.get('status', {}).get('expires', 'Offline')  # Online or offline status
            is_verified = participant.get('verified', False)
            
            features.append({
                'participant_name': full_name,
                'status': status,
                'is_verified': is_verified
            })
    except Exception as e:
        logging.error(f"Error extracting participant features: {e}")
    return features

# Process all_entities data
def process_all_entities(entities):
    structured_data = []
    try:
        for entity in entities:
            entity_type = entity.get('_', '')
            
            if entity_type == "Channel":
                title = entity.get('title', '')
                username = entity.get('username', '')
                structured_data.append({
                    'entity': 'Channel',
                    'type': 'Title',
                    'value': title
                })
                structured_data.append({
                    'entity': 'Channel',
                    'type': 'Username',
                    'value': username
                })
            elif entity_type == "User":
                first_name = entity.get('first_name', '')
                phone = entity.get('phone', '')
                structured_data.append({
                    'entity': 'User',
                    'type': 'First Name',
                    'value': first_name
                })
                structured_data.append({
                    'entity': 'User',
                    'type': 'Phone',
                    'value': phone
                })
    except Exception as e:
        logging.error(f"Error processing all_entities: {e}")
    return structured_data

# Handle NaN or infinite values in the dataset
def handle_null_and_infinite_values(df):
    # Replace NaN with a placeholder string
    df.replace([np.nan, np.inf, -np.inf], 'UNKNOWN', inplace=True)

    # Convert all columns to string format to avoid type issues
    df = df.astype(str)
    return df

# Create processed data folder
def create_processed_data_directory():
    base_folder = "processed_data"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"processed_data_{timestamp}"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Create a consolidated and clean dataset for One-Class SVM
def create_combined_csv(processed_data_folder, combined_messages, combined_iocs, combined_participants, combined_entities):
    try:
        # Create individual DataFrames
        df_messages = pd.DataFrame({'processed_messages': combined_messages})
        df_iocs = pd.DataFrame(combined_iocs)
        df_participants = pd.DataFrame(combined_participants)
        df_entities = pd.DataFrame(combined_entities)

        # Merge all DataFrames
        df_combined = pd.concat([
            df_messages,
            df_iocs,
            df_participants,
            df_entities
        ], axis=1)

        # Clean up the data
        df_combined = handle_null_and_infinite_values(df_combined)

        # Save the consolidated dataset
        combined_csv_path = os.path.join(processed_data_folder, 'merged_data_for_svm.csv')
        df_combined.to_csv(combined_csv_path, index=False)
        logging.info(f"Consolidated dataset saved to {combined_csv_path}")
    except Exception as e:
        logging.error(f"Error creating consolidated CSV: {e}")

# Main processing function
def main():
    paths = locate_files()
    processed_data_folder = create_processed_data_directory()

    # Combined lists for broadcast and direct message data
    combined_messages = []
    combined_iocs = []
    combined_participants = []
    combined_entities = []

    # Process messages from all broadcast channels and direct messages into combined CSV
    for message_file in paths["messages"]:
        messages = load_json(message_file)
        combined_messages.extend(preprocess_messages(messages))
    
    # Process IOCs from all broadcast channels and direct messages into combined CSV
    for ioc_file in paths["iocs"]:
        iocs = load_json(ioc_file)
        combined_iocs.extend(extract_iocs(iocs))

    # Process participants from all broadcast channels and direct messages into combined CSV
    for participant_file in paths["participants"]:
        participants = load_json(participant_file)
        combined_participants.extend(extract_participant_features(participants))

    # Process all entities from all broadcast channels and direct messages into combined CSV
    for entity_file in paths["all_entities"]:
        entities = load_json(entity_file)
        combined_entities.extend(process_all_entities(entities))

    # Save individual CSVs for processed data
    if combined_messages:
        df_messages = pd.DataFrame({'processed_messages': combined_messages})
        df_messages = handle_null_and_infinite_values(df_messages)
        df_messages.to_csv(os.path.join(processed_data_folder, 'processed_messages_combined.csv'), index=False)

    if combined_iocs:
        df_iocs = pd.DataFrame(combined_iocs)
        df_iocs = handle_null_and_infinite_values(df_iocs)
        df_iocs.to_csv(os.path.join(processed_data_folder, 'processed_iocs_combined.csv'), index=False)

    if combined_participants:
        df_participants = pd.DataFrame(combined_participants)
        df_participants = handle_null_and_infinite_values(df_participants)
        df_participants.to_csv(os.path.join(processed_data_folder, 'processed_participants_combined.csv'), index=False)

    if combined_entities:
        df_entities = pd.DataFrame(combined_entities)
        df_entities = handle_null_and_infinite_values(df_entities)
        df_entities.to_csv(os.path.join(processed_data_folder, 'processed_all_entities_combined.csv'), index=False)

    # Create combined CSV
    create_combined_csv(
        processed_data_folder,
        combined_messages,
        combined_iocs,
        combined_participants,
        combined_entities
    )

if __name__ == "__main__":
    main()
