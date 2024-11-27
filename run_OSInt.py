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

def prompt_user_after_script(script_name, is_scrape_script=False):
    """
    Prompts the user after each script for the next action.
    """
    while True:
        print(f"\n{script_name} finished running.")
        if is_scrape_script:
            action = input("What do you want to do next? (1: Restart scrape.py, 2: Run next script, 3: Exit): ")
            if action == "1":
                return "restart"
            elif action == "2":
                return "next"
            elif action == "3":
                return "exit"
        else:
            action = input("What do you want to do next? (1: Run next script, 2: Exit): ")
            if action == "1":
                return "next"
            elif action == "2":
                return "exit"
        print("Invalid input. Please choose a valid option.")

def run_scrape_script():
    """
    Runs the scrape.py script with user-provided or default flags.
    """
    default_command = "--get-messages --get-participants --get-entities --export-to-es"
    
    while True:
        # Ask user for input flags
        print("Enter the command flags for the scrape.py script (default: '--get-messages --get-participants --get-entities --export-to-es').")
        user_command = input("Enter your flags for scrape.py (or press Enter to use defaults): ")
        command_to_run = f"python scrape.py {user_command if user_command else default_command}"
        
        if run_script(command_to_run):
            user_action = prompt_user_after_script("scrape.py", is_scrape_script=True)
            if user_action == "restart":
                continue  # Restart scrape.py with new user input
            elif user_action == "next":
                return True  # Proceed to the next script
            elif user_action == "exit":
                return False  # Exit the process
        else:
            logging.error("scrape.py failed.")
            return False

def run_next_scripts():
    """
    Runs the next set of predefined scripts automatically.
    """
    # List of scripts to execute in sequence
    scripts = [
        "python data_preprocessing.py",       # Preprocess the data
        "python model_analysis_pipeline.py", # Run the model analysis pipeline
        "python anomaly_detection.py"        # Perform anomaly detection
    ]
    
    for script in scripts:
        if not run_script(script):
            logging.error(f"Stopping execution due to failure in {script}.")
            return False

        user_action = prompt_user_after_script(script)
        if user_action == "next":
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
