import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def locate_latest_processed_data():
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
        logging.error("No subfolders found in the 'processed_data' directory.")
        return None

    latest_folder = max(subfolders, key=os.path.getmtime)
    logging.info(f"Located latest processed data folder: {latest_folder}")
    return latest_folder


def load_csv_files(processed_data_folder):
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
            logging.info(f"Loaded '{key}' data from {file_path} with {len(dataframes[key])} records.")
        else:
            logging.warning(f"File '{filename}' not found in {processed_data_folder}. Skipping.")

    return dataframes


def plot_feature_importance(model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Important Features")
    plt.gca().invert_yaxis()
    plt.show()


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def apply_random_forest_with_visualizations(messages_df):
    if "processed_messages" not in messages_df.columns:
        logging.error("'processed_messages' column not found in the messages DataFrame.")
        return

    # Handle missing values
    messages_df = messages_df.dropna(subset=["processed_messages"])
    messages_df = messages_df[messages_df["processed_messages"].str.strip().astype(bool)]

    if messages_df.empty:
        logging.error("No valid messages found after cleaning.")
        return

    # Sample a subset for testing
    messages_df = messages_df.sample(n=1000, random_state=42)  # Reduce to 1000 samples for testing

    # Dummy target variable
    messages_df["label"] = [0 if i % 2 == 0 else 1 for i in range(len(messages_df))]

    X = messages_df["processed_messages"]
    y = messages_df["label"]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)  # Reduced to 1000 features
    X_vectorized = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Random Forest model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=10, n_jobs=-1)  # Reduced to 10 trees
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)

    # Evaluation metrics
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(rf_model, vectorizer)
    plot_roc_curve(y_test, y_proba)


def main():
    processed_data_folder = locate_latest_processed_data()
    if not processed_data_folder:
        return

    dataframes = load_csv_files(processed_data_folder)

    if "messages" in dataframes:
        apply_random_forest_with_visualizations(dataframes["messages"])
    else:
        logging.warning("Messages data not found. Skipping Random Forest application.")


if __name__ == "__main__":
    main()




## ------------------------   NOT TO BE USED FOR TESTING   ------------------------ ##

# def apply_random_forest(messages_df):
#     """
#     Applies a Random Forest model on the processed messages data.

#     Args:
#         messages_df (pd.DataFrame): The processed messages DataFrame.

#     Returns:
#         None
#     """
#     if "processed_messages" not in messages_df.columns:
#         logging.error("'processed_messages' column not found in the messages DataFrame.")
#         return

#     # Handle missing values in the 'processed_messages' column
#     messages_df = messages_df.dropna(subset=["processed_messages"])  # Remove rows with NaN in 'processed_messages'
#     messages_df = messages_df[messages_df["processed_messages"].str.strip().astype(bool)]  # Remove empty strings

#     if messages_df.empty:
#         logging.error("No valid messages found after cleaning. Exiting Random Forest application.")
#         return

#     # Create a dummy target variable for demonstration (binary classification)
#     # In a real scenario, replace this with actual labels
#     messages_df["label"] = [0 if i % 2 == 0 else 1 for i in range(len(messages_df))]

#     # Split data into features (X) and labels (y)
#     X = messages_df["processed_messages"]
#     y = messages_df["label"]

#     # Convert text data into numerical form using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words="english")
#     X_vectorized = vectorizer.fit_transform(X)

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_vectorized, y, test_size=0.2, random_state=42
#     )

#     # Initialize and train the Random Forest model
#     rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
#     rf_model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = rf_model.predict(X_test)

#     # Evaluate the model
#     logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
#     logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")