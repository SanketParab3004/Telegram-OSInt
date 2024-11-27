import os
import json
from pathlib import Path

def find_latest_folder(base_dir='output'):
    # Get all folders in the output directory sorted by creation time
    folders = [f for f in Path(base_dir).iterdir() if f.is_dir()]
    latest_folder = max(folders, key=os.path.getctime)
    return latest_folder

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def locate_files():
    latest_folder = find_latest_folder()
    paths = {
        "all_entities": latest_folder / "all_entities.json",
        "messages": None,
        "iocs": None,
        "participants": None,
    }
    
    # Locate files within subdirectories
    for subfolder in latest_folder.iterdir():
        if subfolder.is_dir():
            if "broadcast_channel" in subfolder.name:
                paths["messages"] = next(subfolder.glob("messages_*.json"), None)
                paths["iocs"] = next(subfolder.glob("iocs_*.json"), None)
            elif "direct_message" in subfolder.name:
                paths["participants"] = next(subfolder.glob("participants_*.json"), None)

    return paths

# Example Usage
if __name__ == "__main__":
    file_paths = locate_files()
    print("Located JSON files:", file_paths)
