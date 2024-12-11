import os
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
REPORT_FILENAME = "threat_report.txt"
PREDICTION_FOLDER = "Threats"  # Base folder where the prediction CSV is saved

def generate_report(predictions_df, report_folder):
    """
    Generate and save a comprehensive report.
    """
    logging.info("Generating report...")

    # Basic report
    report = []

    # Total number of records
    total_records = len(predictions_df)
    report.append(f"Total records: {total_records}\n")

    # Total number of threats predicted
    total_threats = predictions_df['Threat_Prediction'].sum()
    report.append(f"Total threats predicted: {total_threats}\n")

    # Threat percentage
    threat_percentage = (total_threats / total_records) * 100
    report.append(f"Threat percentage: {threat_percentage:.2f}%\n")

    # Distribution of threats and non-threats
    threat_distribution = predictions_df['Threat_Prediction'].value_counts()
    report.append(f"Threat distribution:\n{threat_distribution}\n")

    # Performance metrics: Accuracy, Confusion Matrix, etc.
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_true = predictions_df['label']  # Assuming the labels are present in the CSV
    y_pred = predictions_df['Threat_Prediction']
    
    accuracy = accuracy_score(y_true, y_pred)
    report.append(f"Accuracy: {accuracy:.4f}\n")
    
    report.append("Classification Report:\n")
    report.append(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    report.append(f"Confusion Matrix:\n{cm}\n")

    # Save the report to a text file in the same folder as predictions
    report_file_path = os.path.join(report_folder, REPORT_FILENAME)
    with open(report_file_path, 'w') as f:
        f.writelines(report)
    
    logging.info(f"Report saved to {report_file_path}")


def main():
    try:
        # Locate the latest prediction data folder (same folder as threat predictions)
        latest_folder = max(
            [os.path.join(PREDICTION_FOLDER, folder) for folder in os.listdir(PREDICTION_FOLDER)],
            key=os.path.getmtime
        )
        
        predictions_file = os.path.join(latest_folder, "threat_predictions.csv")
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Prediction file not found: {predictions_file}")

        # Load predictions data
        predictions_df = pd.read_csv(predictions_file)
        logging.info(f"Loaded predictions data from {predictions_file}.")

        # Generate and save the report in the same folder as predictions
        generate_report(predictions_df, latest_folder)

    except Exception as e:
        logging.error(f"An error occurred while generating the report: {e}")


if __name__ == "__main__":
    main()
