from pathlib import Path
from rag_project.logger import logger
from rag_project.exceptions import DocumentLoadingError
from rag_project.utils import load_config
import json
import os
import warnings
warnings.filterwarnings('ignore')

def load_documents_from_folder():

    config = load_config()
    folder_path = config['paths']['raw_data']
    
    folder = Path(folder_path)
    try:
        with open(os.path.join(folder_path,"project_1_publications.json"), 'r') as file:
            data = json.load(file)

        logger.info(f"Data Loaded {len(data)}")
            
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise DocumentLoadingError("Failed to load json file.") from e

    if not data:
        raise DocumentLoadingError("No documents loaded. Check the folder path and file types.")

    return data
