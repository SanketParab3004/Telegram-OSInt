# Telegram OSINT for Cyber Threat Intelligence Analysis

Welcome to the **Telegram OSINT for Cyber Threat Intelligence Analysis** project. This tool automates the collection, processing, and analysis of Telegram data for OSINT purposes, focusing on cybersecurity insights.

**To view more information about the project, visit the [Wiki](https://github.com/SanketParab3004/Telegram-OSInt/wiki)!**

---

## **Features**
- **Data Collection**: Scrape Telegram messages, participants, and entities with flexible configurations.
- **Preprocessing**: Clean and transform data for seamless analysis.
- **Analysis**: Leverage machine learning pipelines for threat intelligence.
- **Anomaly Detection**: Identify potential threats or unusual activities.
- **Customizable Workflow**: Run scripts sequentially with user inputs through the master script.
- **Integration**: Export data to Elasticsearch and analyze through Kibana.

---

## **Quick Setup**

### **Recommended**  
Visit the Wiki’s **[Set Up Development Environment](https://github.com/SanketParab3004/Telegram-OSInt/wiki/Set-Up-Development-Environment)** page for detailed setup instructions.

### **Installations**

1. **Git**: [Download and install Git](https://git-scm.com/downloads).  
2. **Python 3.11 or lower**: [Download Python](https://www.python.org/downloads/).  
3. **SQLite3**:  
   - **Windows**: [Download SQLite3](https://www.sqlite.org/download.html).  
   - **Linux**: Install via `sudo apt install sqlite3`.  
   - **MacOS**: Pre-installed.  
4. **Elasticsearch & Kibana**:  
   - [Download Elasticsearch](https://www.elastic.co/downloads/elasticsearch).  
   - [Download Kibana](https://www.elastic.co/downloads/kibana).

### **Telegram Installation / API Setup**

> **Note**: Use a burner phone number, burner email, and a VM for Telegram setup. For OPSEC recommendations, refer to the [Wiki](https://github.com/SanketParab3004/Telegram-OSInt/wiki).

1. Install Telegram Desktop and set up a Telegram account.
2. Visit [Telegram API Development Tools](https://my.telegram.org/auth) and log in.
3. Create an application:
   - **App Title**: Any name.
   - **URL**: www.telegram.org.
   - **Platform**: Desktop.
   - **Description**: Any description.
4. Save your **App api_id** and **App api_hash**.

---

## **Environment Setup**

### **Step 1: Create a Virtual Environment**
```bash
python -m venv venv
source venv/Scripts/activate  # For Windows (Git Bash)
source venv/bin/activate      # For UNIX-based systems
```

To deactivate:
```bash
deactivate
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Optional**: Install individual dependencies if needed:
```bash
pip install telethon argostranslate lingua-language-detector requests elasticsearch ijson
```

### **Step 3: Configure API Credentials**
Create a `configs.py` file with the following template:
```python
PHONE_NUMBER = "+1234567890"  # Replace with your phone number.
API_HASH = "your_api_hash"   # Replace with your Telegram API hash.
API_ID = 123456              # Replace with your Telegram API ID.

# Optional proxy configuration
PROXIES = None  # Replace with proxy details if needed.

# Elasticsearch configuration (optional)
es_username = None
es_password = None
es_ca_cert_path = None
```

---

## **Usage**

### **Run the Master Script**
```bash
python run_OSInt.py
```

Follow the prompts to:
- Scrape Telegram data using `scrape.py`.
- Preprocess data with `data_preprocessing.py`.
- Perform analysis with `model_analysis_pipeline.py`.
- Detect anomalies using `anomaly_detection.py`.

### **Individual Script Execution**
You can also run scripts separately:
```bash
python scrape.py --get-messages --get-participants --export-to-es
python data_preprocessing.py
python model_analysis_pipeline.py
python anomaly_detection.py
```

---

## **Additional Resources**
- **[Wiki](https://github.com/SanketParab3004/Telegram-OSInt/wiki/Telegram-OSINT-Research-Environment-Setup)**: For setup, configuration, and OPSEC details.
- **Elasticsearch & Kibana**: Visualize exported data.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.