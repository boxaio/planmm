import sys
import os
import re
from pathlib import Path
from datetime import datetime


def get_file_absolute_paths(folder_path):
    file_paths = []
    
    if not os.path.isdir(folder_path):
        print(f"Error: folder '{folder_path}' not exist.")
        return file_paths
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(file_path)
    
    return file_paths


def find_latest_timestamp_folder(directory):
    if not os.path.exists(directory):
        return None
    # YYYY-MM-DD_HH-MM-SS
    pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
    
    latest_time = None
    latest_folder = None
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isdir(item_path) and re.match(pattern, item):
            try:
                folder_time = datetime.strptime(item, "%Y-%m-%d_%H-%M-%S")
                
                if latest_time is None or folder_time > latest_time:
                    latest_time = folder_time
                    latest_folder = item_path
            except ValueError:
                continue
    
    return latest_folder

def find_folders_between_time(root_dir, start_time, end_time):
    # YYYY-MM-DD_HH-MM-SS
    time_format = "%Y-%m-%d_%H-%M-%S"
    matched_folders = []
    
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        
        if os.path.isdir(entry_path):
            try:  
                folder_time = datetime.strptime(entry, time_format)
                if (folder_time > start_time) and (folder_time < end_time):
                    matched_folders.append(entry_path)
                    
            except ValueError:
                continue
    
    return matched_folders


def class_from_str(str, module=None, none_on_fail=False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")


def get_path_to_assets() -> Path:
    return Path("/media/box/Elements/MyExp/TMead/") / "assets"


def get_path_to_externals() -> Path:
    return Path("/media/box/Elements/MyExp/TMead/") / "external"


def get_path_to_tracker() -> Path:
    return Path("/media/box/Elements/MyExp/TMead/") / "tracker"