from roboflow import Roboflow
import os
import shutil

def download_data(api_key, target_folder, workspace_name, project_name, project_version, dtype):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(project_version)
    
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Download the dataset
    dataset = version.download(dtype)
    
    # Extract the path where the dataset was downloaded
    download_folder = dataset.location
    
    # Move contents to the target folder
    for item in os.listdir(download_folder):
        s = os.path.join(download_folder, item)
        d = os.path.join(target_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)  
        else:
            shutil.copy2(s, d)
    
    # Remove the original download folder
    shutil.rmtree(download_folder)
