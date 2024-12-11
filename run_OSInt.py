import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(command):
    """
    Runs a given script using the subprocess module.
    """
    logging.info(f"Running command: {command}")
    try:
        subprocess.run(command, check=True, shell=True)
        logging.info(f"{command} executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while running {command}: {e}")
        return False
    return True

def prompt_user(script_name):
    """
    Prompts the user with options for the current script.
    """
    while True:
        print(f"\nDo you want to run {script_name}?")
        action = input(f"(1: Yes, 2: Skip, 3: Exit): ")
        if action == "1":
            return "run"
        elif action == "2":
            return "skip"
        elif action == "3":
            return "exit"
        print("Invalid input. Please choose a valid option.")

def run_scrape_script():
    """
    Runs the scrape.py script with user-provided or default flags.
    """
    default_command = "--get-messages --get-participants --get-entities --export-to-es"
    
    user_action = prompt_user("scrape.py")
    if user_action == "skip":
        return True  # Skip scrape.py and proceed to the next script
    elif user_action == "exit":
        return False  # Exit the process

    while user_action == "run":
        # Ask user for input flags
        print("Enter the command flags for the scrape.py script (default: '--get-messages --get-participants --get-entities --export-to-es').")
        user_command = input("Enter your flags for scrape.py (or press Enter to use defaults): ")
        command_to_run = f"python scrape.py {user_command if user_command else default_command}"
        
        if run_script(command_to_run):
            user_action = prompt_user("scrape.py")
            if user_action == "run":
                continue  # Restart scrape.py with new user input
            elif user_action == "skip":
                return True  # Skip to the next script
            elif user_action == "exit":
                return False  # Exit the process
        else:
            logging.error("scrape.py failed.")
            return False

def run_next_scripts():
    """
    Runs the next set of predefined scripts automatically with user prompts.
    """
    # List of scripts to execute in sequence
    scripts = [
        "python data_preprocessing.py",       # Preprocess the data
        "python model_analysis_pipeline.py", # Run the model analysis pipeline
        "python anomaly_detection.py",        # Perform anomaly detection
        "python predict_threats.py",         # Predict threats
        "python threat_clustering.py",       # Cluster threats
        "python generate_report.py"          # Generate report
    ]
    
    for script in scripts:
        user_action = prompt_user(script)
        if user_action == "run":
            if not run_script(script):
                logging.error(f"Stopping execution due to failure in {script}.")
                return False
        elif user_action == "skip":
            logging.info(f"Skipping {script}.")
            continue
        elif user_action == "exit":
            return False
    return True

def main():
    print("Welcome to the Telegram OSINT Master Script")
    
    # Step 1: Run the scrape.py script with user input
    if run_scrape_script():
        # Step 2: Run the next scripts automatically
        if run_next_scripts():
            logging.info("All scripts executed successfully.")
        else:
            logging.error("Some scripts failed or were stopped by the user.")
    else:
        logging.error("scrape.py failed or was stopped by the user.")

if __name__ == "__main__":
    main()
