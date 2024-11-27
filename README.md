# Telegram OSINT for Cyber Threat Intelligence Analysis

This project is designed to automate the collection, preprocessing, analysis, and anomaly detection of data from Telegram groups and channels. It facilitates Open Source Intelligence (OSINT) workflows with modular Python scripts to scrape and analyze data for cybersecurity insights.

## **Features**
- **Data Scraping**: Collect messages, participants, and entities from Telegram using `scrape.py` with customizable flags.
- **Data Preprocessing**: Clean, transform, and prepare raw data for analysis.
- **Model Analysis Pipeline**: Use machine learning models to analyze patterns and extract meaningful insights.
- **Anomaly Detection**: Identify anomalies in the collected data that could signify threats or unusual activities.
- **Custom Workflow**: The master script allows sequential execution of all modules with user input and logging for better usability.

---

## **Installation and Setup**

### **1. Prerequisites**
- Python 3.8+ installed.
- Access to Telegram API credentials (`api_id` and `api_hash`). Obtain them from [Telegram's Developer Portal](https://my.telegram.org/auth).
- Elasticsearch (if using the export feature in `scrape.py`).

### **2. Clone the Repository**
```bash
git clone https://github.com/SanketParab3004/Telegram-OSInt.git
cd Telegram-OSInt
```

### **3. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **5. Configure Telegram API**
Create a `config.py` file in the project root with the following structure:
```python
API_ID = "your_api_id"
API_HASH = "your_api_hash"
SESSION_NAME = "your_session_name"
```

### **6. Elasticsearch Setup (Optional)**
If you plan to export data to Elasticsearch:
1. Install and configure Elasticsearch.
2. Update the `scrape.py` script's export configuration if required.

---

## **Usage**

### **Run the Master Script**
The master script orchestrates the execution of all other scripts.
```bash
python master_script.py
```

Follow the prompts to:
- Scrape data with `scrape.py` (customizable flags provided in the prompts).
- Preprocess scraped data with `data_preprocessing.py`.
- Analyze data with `model_analysis_pipeline.py`.
- Perform anomaly detection using `anomaly_detection.py`.

---

## **Scripts Overview**

### **1. `scrape.py`**
- Collects Telegram data such as messages, participants, and entities.
- Exports data to Elasticsearch if the `--export-to-es` flag is used.

### **2. `data_preprocessing.py`**
- Cleans and preprocesses raw data for downstream analysis.

### **3. `model_analysis_pipeline.py`**
- Implements machine learning models to analyze Telegram data and generate insights.

### **4. `anomaly_detection.py`**
- Detects anomalies in the data, potentially highlighting cybersecurity threats.

### **5. `master_script.py`**
- Interactive script to run all modules in sequence with user input.

---

## **Flags for `scrape.py`**
- `--get-messages`: Fetch messages from Telegram.
- `--get-participants`: Retrieve participants of the group or channel.
- `--get-entities`: Collect entities mentioned in messages.
- `--export-to-es`: Export collected data to Elasticsearch.

---

## **Contribution**
Feel free to submit issues or pull requests to improve the project. For significant contributions, please discuss the changes beforehand.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.
