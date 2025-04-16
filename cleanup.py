import os
import shutil
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_repository(backup_dir='backup'):
    """
    Clean up the repository by moving unused files to a backup folder.
    
    Args:
        backup_dir: Directory to move unused files to
    """
    # Create backup directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{backup_dir}_{timestamp}"
    os.makedirs(backup_path, exist_ok=True)
    
    logger.info(f"Cleaning up repository. Unused files will be moved to {backup_path}")
    
    # Files to keep (essential files)
    keep_files = [
        # Core implementation
        'src/multimodal_mcts_qa.py',
        'src/mcts.py',
        'src/models/__init__.py',
        'src/models/llava_model.py',
        'src/models/llama_react.py',
        'src/tools/__init__.py',
        'src/tools/ocr_tool.py',
        'src/data/scienceqa_dataset.py',
        'src/data/__init__.py',
        'src/__init__.py',
        'src/mcts/__init__.py',
        'src/mcts/mcts.py',
        'src/mcts/node.py',
        'src/mcts/multimodal_qa_node.py',
        
        # Scripts
        'evaluate_scienceqa.py',
        'cleanup.py',
        'scripts/download_models.py',
        
        # Config files
        'requirements.txt',
        'setup.py',
        'README.md',
    ]
    
    # Include all directories in data/scienceqa
    scienceqa_data_dir = 'data/scienceqa'
    if os.path.exists(scienceqa_data_dir):
        for root, dirs, files in os.walk(scienceqa_data_dir):
            for file in files:
                rel_path = os.path.join(root, file).replace('\\', '/')
                keep_files.append(rel_path)
    
    # Move files that are not in the keep_files list to backup
    moved_files = []
    for root, dirs, files in os.walk('.'):
        # Skip backup directory and .git
        if root.startswith(f'./{backup_dir}') or root.startswith('./.git'):
            continue
            
        for file in files:
            file_path = os.path.join(root, file).replace('\\', '/')
            # Remove leading ./ if present
            if file_path.startswith('./'):
                file_path = file_path[2:]
                
            # Skip files in the keep list
            if file_path in keep_files:
                continue
                
            # Skip certain file types
            if file_path.endswith('.git') or file_path.endswith('.gitignore'):
                continue
                
            # Create the backup directory structure
            backup_file_dir = os.path.join(backup_path, os.path.dirname(file_path))
            os.makedirs(backup_file_dir, exist_ok=True)
            
            # Move the file
            try:
                backup_file_path = os.path.join(backup_path, file_path)
                shutil.move(file_path, backup_file_path)
                moved_files.append(file_path)
                logger.info(f"Moved {file_path} to {backup_file_path}")
            except Exception as e:
                logger.error(f"Error moving {file_path}: {e}")
    
    # Print summary
    logger.info(f"Cleanup complete. Moved {len(moved_files)} files to {backup_path}")
    
    return moved_files, backup_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up repository')
    parser.add_argument('--backup_dir', type=str, default='backup',
                        help='Directory to move unused files to')
    
    args = parser.parse_args()
    
    moved_files, backup_path = cleanup_repository(backup_dir=args.backup_dir) 