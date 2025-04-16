import os
import argparse
import logging
import requests
import zipfile
import tarfile
from tqdm import tqdm
import huggingface_hub
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url, target_path):
    """
    Download a file with a progress bar.
    
    Args:
        url: URL to download
        target_path: Path to save the file
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if os.path.exists(target_path):
        logger.info(f"File already exists: {target_path}")
        return
    
    logger.info(f"Downloading {url} to {target_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as file, tqdm(
        desc=os.path.basename(target_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_archive(archive_path, extract_path):
    """
    Extract a zip or tar archive.
    
    Args:
        archive_path: Path to the archive
        extract_path: Path to extract to
    """
    os.makedirs(extract_path, exist_ok=True)
    
    logger.info(f"Extracting {archive_path} to {extract_path}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            tar_ref.extractall(extract_path)
    else:
        logger.warning(f"Unsupported archive format: {archive_path}")

def download_hf_model(model_name, target_dir):
    """
    Download a Hugging Face model.
    
    Args:
        model_name: Name of the model on Hugging Face
        target_dir: Directory to save the model
    """
    logger.info(f"Downloading model {model_name} to {target_dir}")
    
    try:
        huggingface_hub.snapshot_download(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded {model_name}")
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {e}")

def download_scienceqa():
    """Download ScienceQA dataset from GitHub."""
    DATA_DIR = "data/scienceqa"
    GITHUB_URL = "https://github.com/lupantech/ScienceQA/archive/refs/heads/main.zip"
    ARCHIVE_PATH = f"{DATA_DIR}/scienceqa_github.zip"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download the dataset archive
    try:
        logger.info(f"Downloading ScienceQA dataset from GitHub")
        download_file(GITHUB_URL, ARCHIVE_PATH)
        
        # Extract the archive
        extract_archive(ARCHIVE_PATH, DATA_DIR)
        
        # Move files from extracted directory
        extracted_dir = os.path.join(DATA_DIR, "ScienceQA-main")
        if os.path.exists(extracted_dir):
            # Move data files
            data_dir = os.path.join(extracted_dir, "data")
            if os.path.exists(data_dir):
                for item in os.listdir(data_dir):
                    src = os.path.join(data_dir, item)
                    dst = os.path.join(DATA_DIR, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            
            # Move images directory
            images_dir = os.path.join(extracted_dir, "images")
            if os.path.exists(images_dir):
                dst_images = os.path.join(DATA_DIR, "images")
                if os.path.exists(dst_images):
                    shutil.rmtree(dst_images)
                shutil.copytree(images_dir, dst_images)
        
        logger.info(f"ScienceQA dataset downloaded and extracted to {DATA_DIR}")
    except Exception as e:
        logger.error(f"Error downloading or extracting ScienceQA dataset: {e}")

def download_models(models):
    """
    Download specified models.
    
    Args:
        models: List of model names to download
    """
    for model_name in models:
        target_dir = f"models/{model_name.replace('/', '_')}"
        download_hf_model(model_name, target_dir)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download models and datasets")
    
    parser.add_argument(
        "--download_scienceqa", 
        action="store_true",
        help="Download ScienceQA dataset"
    )
    
    parser.add_argument(
        "--download_llava", 
        action="store_true",
        help="Download LLaVA model"
    )
    
    parser.add_argument(
        "--download_llama", 
        action="store_true",
        help="Download LLaMA model"
    )
    
    parser.add_argument(
        "--llava_model", 
        type=str, 
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA model to download"
    )
    
    parser.add_argument(
        "--llama_model", 
        type=str, 
        default="meta-llama/Llama-2-7b-chat-hf",
        help="LLaMA model to download"
    )
    
    parser.add_argument(
        "--download_all", 
        action="store_true",
        help="Download all models and datasets"
    )
    
    parser.add_argument(
        "--download_data_only", 
        action="store_true",
        help="Download only the datasets, not the models"
    )
    
    parser.add_argument(
        "--download_models_only", 
        action="store_true",
        help="Download only the models, not the datasets"
    )
    
    args = parser.parse_args()
    
    # Handle special flags
    if args.download_data_only:
        args.download_scienceqa = True
        args.download_llava = False
        args.download_llama = False
    
    if args.download_models_only:
        args.download_scienceqa = False
        args.download_llava = True
        args.download_llama = True
    
    # Download ScienceQA dataset
    if args.download_scienceqa or args.download_all:
        logger.info("Downloading ScienceQA dataset")
        download_scienceqa()
    
    # Download models
    models_to_download = []
    
    if args.download_llava or args.download_all:
        models_to_download.append(args.llava_model)
    
    if args.download_llama or args.download_all:
        models_to_download.append(args.llama_model)
    
    if models_to_download:
        logger.info(f"Downloading models: {models_to_download}")
        download_models(models_to_download)
    
    logger.info("Download completed")

if __name__ == "__main__":
    main() 