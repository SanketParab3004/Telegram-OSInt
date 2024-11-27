import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_scrape_script(user_command):
    """
    Runs the scrape.py script with the user-provided flags.
    """
    command = f"python scrape.py {user_command}"
    logging.info(f"Running command: {command}")
    try:
        subprocess.run(command, check=True, shell=True)
        logging.info("scrape.py execution completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while running scrape.py: {e}")
        return False
    return True

def run_next_scripts():
    """
    Runs the next set of predefined scripts automatically after scrape.py is done.
    """
    # Define the list of scripts to run after scrape.py
    scripts = [
        "python script1.py",  # Example script 1
        "python script2.py",  # Example script 2
        "python script3.py"   # Example script 3
    ]
    
    for script in scripts:
        try:
            logging.info(f"Running {script}...")
            subprocess.run(script, check=True, shell=True)
            logging.info(f"{script} executed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while running {script}: {e}")

def main():
    print("Welcome to the Master Script")
    
    # Ask user for scrape.py flags input
    print("Enter the command flags for the scrape.py script (e.g. --get-messages --get-participants --get-entities --export-to-es).")
    user_command = input("Enter your flags for scrape.py: ")

    if not user_command:
        logging.warning("No flags entered. Please enter valid flags for scrape.py.")
        return
    
    # Step 1: Run the scrape.py script with the provided flags
    if run_scrape_script(user_command):
        # Step 2: Run the next scripts automatically
        run_next_scripts()

if __name__ == "__main__":
    main()
