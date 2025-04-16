#!/usr/bin/env python
# Script to download and set up the ScienceQA dataset

import os
import sys
import logging
import requests
import zipfile
import shutil
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ScienceQA dataset URLs
SCIENCEQA_URL = "https://github.com/lupantech/ScienceQA/archive/refs/heads/main.zip"
SCIENCEQA_DATA_URL = "https://sci-qa.s3.us-west-1.amazonaws.com/data.zip"
SCIENCEQA_IMAGES_URL = "https://sci-qa.s3.us-west-1.amazonaws.com/images.zip"

# Destination paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SCIENCEQA_DIR = os.path.join(DATA_DIR, "scienceqa")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

def download_file(url, dest_path):
    """
    Download a file from a URL with progress bar.
    """
    logger.info(f"Downloading {url} to {dest_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a directory.
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_to)

def setup_scienceqa():
    """
    Download and set up the ScienceQA dataset.
    """
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SCIENCEQA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Download ScienceQA repository
    repo_zip_path = os.path.join(TEMP_DIR, "scienceqa_repo.zip")
    if not os.path.exists(repo_zip_path):
        download_file(SCIENCEQA_URL, repo_zip_path)
    
    # Extract repository
    extract_zip(repo_zip_path, TEMP_DIR)
    
    # Move main files to the right location
    repo_dir = os.path.join(TEMP_DIR, "ScienceQA-main")
    if os.path.exists(repo_dir):
        logger.info("Copying ScienceQA repository files")
        # Copy key files
        for item in os.listdir(repo_dir):
            src = os.path.join(repo_dir, item)
            dest = os.path.join(SCIENCEQA_DIR, item)
            if os.path.isdir(src):
                if not os.path.exists(dest):
                    shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
    
    # Download data.zip
    data_zip_path = os.path.join(TEMP_DIR, "data.zip")
    if not os.path.exists(data_zip_path):
        download_file(SCIENCEQA_DATA_URL, data_zip_path)
    
    # Extract data
    extract_zip(data_zip_path, SCIENCEQA_DIR)
    
    # Download images.zip
    images_zip_path = os.path.join(TEMP_DIR, "images.zip")
    if not os.path.exists(images_zip_path):
        download_file(SCIENCEQA_IMAGES_URL, images_zip_path)
    
    # Extract images
    extract_zip(images_zip_path, SCIENCEQA_DIR)
    
    # Clean up
    logger.info("Cleaning up temporary files")
    # Keep the zip files but remove extracted temp files
    if os.path.exists(os.path.join(TEMP_DIR, "ScienceQA-main")):
        shutil.rmtree(os.path.join(TEMP_DIR, "ScienceQA-main"))
    
    logger.info("ScienceQA dataset setup complete!")
    logger.info(f"Dataset location: {SCIENCEQA_DIR}")

if __name__ == "__main__":
    logger.info("Starting ScienceQA dataset download and setup")
    setup_scienceqa() 