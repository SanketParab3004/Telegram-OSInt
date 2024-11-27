import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def locate_latest_processed_data():
    """
    Locates the most recently created dynamic folder inside the 'processed_data' directory.
    
    Returns:
        The path to the latest dynamic folder.
    """
    processed_data_base = "processed_data"
    if not os.path.exists(processed_data_base):
        logging.error(f"Processed data folder '{processed_data_base}' does not exist.")
        return None

    subfolders = [
        os.path.join(processed_data_base, folder)
        for folder in os.listdir(processed_data_base)
        if os.path.isdir(os.path.join(processed_data_base, folder))
    ]

    if not subfolders:
        logging.error("No subfolders found in the processed_data directory.")
        return None

    latest_folder = max(subfolders, key=os.path.getmtime)
    logging.info(f"Located latest processed data folder: {latest_folder}")
    return latest_folder


def load_csv_files(processed_data_folder):
    """
    Loads the CSV files from the given processed data folder.
    
    Args:
        processed_data_folder: The path to the folder containing processed CSV files.
    
    Returns:
        A dictionary with loaded DataFrames.
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
            logging.info(f"Loaded {key} data from {file_path} with {len(dataframes[key])} records.")
        else:
            logging.warning(f"{filename} not found in {processed_data_folder}. Skipping.")

    return dataframes


def generate_visualizations(dataframes):
    """
    Generates visualizations based on the loaded dataframes.
    
    Args:
        dataframes: A dictionary containing the loaded DataFrames.
    """
    if "messages" in dataframes:
        df_messages = dataframes["messages"]
        # Example: Plot message lengths distribution
        df_messages["message_length"] = df_messages["processed_messages"].str.len()
        plt.figure(figsize=(10, 6))
        sns.histplot(df_messages["message_length"], kde=True, bins=30, color="blue")
        plt.title("Distribution of Message Lengths")
        plt.xlabel("Message Length")
        plt.ylabel("Frequency")
        plt.savefig("message_lengths_distribution.png")
        plt.close()
        logging.info("Saved 'message_lengths_distribution.png'.")

    if "iocs" in dataframes:
        df_iocs = dataframes["iocs"]
        # Example: Count the types of IOCs
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_iocs, x="ioc_type", palette="viridis", order=df_iocs["ioc_type"].value_counts().index)
        plt.title("Frequency of IOC Types")
        plt.xlabel("IOC Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.savefig("ioc_types_frequency.png")
        plt.close()
        logging.info("Saved 'ioc_types_frequency.png'.")

    if "participants" in dataframes:
        df_participants = dataframes["participants"]
        # Example: Plot verification status
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_participants, x="is_verified", palette="coolwarm")
        plt.title("Verification Status of Participants")
        plt.xlabel("Is Verified")
        plt.ylabel("Count")
        plt.savefig("participant_verification_status.png")
        plt.close()
        logging.info("Saved 'participant_verification_status.png'.")


def generate_insights(dataframes):
    """
    Generates basic insights and prints them to the console.
    
    Args:
        dataframes: A dictionary containing the loaded DataFrames.
    """
    if "messages" in dataframes:
        df_messages = dataframes["messages"]
        avg_message_length = df_messages["processed_messages"].str.len().mean()
        logging.info(f"Average message length: {avg_message_length:.2f} characters.")

    if "iocs" in dataframes:
        df_iocs = dataframes["iocs"]
        ioc_counts = df_iocs["ioc_type"].value_counts()
        logging.info("Top IOC Types:")
        logging.info(ioc_counts)

    if "participants" in dataframes:
        df_participants = dataframes["participants"]
        verified_count = df_participants["is_verified"].sum()
        total_participants = len(df_participants)
        logging.info(f"Verified Participants: {verified_count}/{total_participants}.")


def main():
    # Step 1: Locate the latest processed data folder
    processed_data_folder = locate_latest_processed_data()
    if not processed_data_folder:
        return

    # Step 2: Load the CSV files
    dataframes = load_csv_files(processed_data_folder)

    # Step 3: Generate visualizations
    generate_visualizations(dataframes)

    # Step 4: Generate insights
    generate_insights(dataframes)

    logging.info("Data visualization and insights generation complete!")


if __name__ == "__main__":
    main()
