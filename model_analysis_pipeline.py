import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
from datetime import datetime

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


def preprocess_data(messages_df):
    if "processed_messages" not in messages_df.columns:
        logging.error("'processed_messages' column not found in the messages DataFrame.")
        return None, None, None

    messages_df = messages_df.dropna(subset=["processed_messages"])
    messages_df = messages_df[messages_df["processed_messages"].str.strip().astype(bool)]

    if messages_df.empty:
        logging.error("No valid messages found after cleaning.")
        return None, None, None

    messages_df = messages_df.sample(n=5000, random_state=42)  # Sample for quicker processing
    messages_df["label"] = [0 if i % 2 == 0 else 1 for i in range(len(messages_df))]

    X = messages_df["processed_messages"]
    y = messages_df["label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X_vectorized = vectorizer.fit_transform(X)

    return X_vectorized, y, vectorizer


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, vectorizer=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "precision_recall_curve": None,
        "average_precision_score": None,
        "roc_curve": None,
        "auc": None,
    }

    if y_proba is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics["precision_recall_curve"] = (precision, recall)
        metrics["average_precision_score"] = average_precision_score(y_test, y_proba[:, 1])
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics["roc_curve"] = (fpr, tpr)
        metrics["auc"] = auc(fpr, tpr)

    # Print model metrics to the terminal
    print(f"{model_name} Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Classification Report:")
    print(metrics["classification_report"])
    return metrics


def save_plot(fig, plot_name, main_folder):
    folder_path = os.path.join(main_folder, "graphs")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{plot_name}.png")
    fig.savefig(file_path)
    logging.info(f"Figure saved to {file_path}")


def save_classification_report(report, model_name, main_folder):
    # Create a text file with the classification report
    report_path = os.path.join(main_folder, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logging.info(f"Classification report saved to {report_path}")


def plot_feature_importance(model, vectorizer, main_folder):
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1][:20]  # Top 20 features

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Important Features")
    plt.gca().invert_yaxis()
    save_plot(plt, "feature_importance", main_folder)
    plt.show()


def plot_confusion_matrix_comparison(metrics_rf, metrics_svm, main_folder):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(metrics_rf["confusion_matrix"], display_labels=[0, 1]).plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title("Random Forest Confusion Matrix")
    axes[0].grid(False)

    ConfusionMatrixDisplay(metrics_svm["confusion_matrix"], display_labels=[0, 1]).plot(ax=axes[1], cmap=plt.cm.Blues)
    axes[1].set_title("SVM Confusion Matrix")
    axes[1].grid(False)

    plt.tight_layout()
    save_plot(fig, "confusion_matrix_comparison", main_folder)
    plt.show()


def plot_roc_curve_comparison(metrics_rf, metrics_svm, main_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_rf, tpr_rf = metrics_rf["roc_curve"]
    fpr_svm, tpr_svm = metrics_svm["roc_curve"]

    ax.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f"Random Forest (AUC = {metrics_rf['auc']:.2f})")
    ax.plot(fpr_svm, tpr_svm, color='green', lw=2, label=f"SVM (AUC = {metrics_svm['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random guessing")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    save_plot(fig, "roc_curve_comparison", main_folder)
    plt.show()


def plot_precision_recall_curve_comparison(metrics_rf, metrics_svm, main_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_rf, recall_rf = metrics_rf["precision_recall_curve"]
    precision_svm, recall_svm = metrics_svm["precision_recall_curve"]

    ax.plot(recall_rf, precision_rf, color='blue', lw=2, label=f"Random Forest (AP = {metrics_rf['average_precision_score']:.2f})")
    ax.plot(recall_svm, precision_svm, color='green', lw=2, label=f"SVM (AP = {metrics_svm['average_precision_score']:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="lower left")
    save_plot(fig, "precision_recall_curve_comparison", main_folder)
    plt.show()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_folder = os.path.join("model_outputs", f"results_{timestamp}")
    os.makedirs(main_folder, exist_ok=True)

    processed_data_folder = locate_latest_processed_data()
    if not processed_data_folder:
        return

    dataframes = load_csv_files(processed_data_folder)

    if "messages" in dataframes:
        X, y, vectorizer = preprocess_data(dataframes["messages"])
        if X is None or y is None:
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        metrics_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest", vectorizer)
        save_classification_report(metrics_rf["classification_report"], "Random Forest", main_folder)
        
        # SVM
        svm_model = SVC(kernel="linear", probability=True, random_state=42)
        metrics_svm = evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM", vectorizer)
        save_classification_report(metrics_svm["classification_report"], "SVM", main_folder)

        # Compare Results
        plot_confusion_matrix_comparison(metrics_rf, metrics_svm, main_folder)
        plot_roc_curve_comparison(metrics_rf, metrics_svm, main_folder)
        plot_precision_recall_curve_comparison(metrics_rf, metrics_svm, main_folder)
        plot_feature_importance(rf_model, vectorizer, main_folder)

    else:
        logging.warning("Messages data not found. Skipping model applications.")


if __name__ == "__main__":
    main()
