import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from PIL import Image
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

class ScienceQADataset:
    """
    Utility for loading and processing the ScienceQA dataset.
    """
    
    def __init__(
        self, 
        data_dir: str,
        split: str = "train",
        image_dir: Optional[str] = None,
        load_images: bool = True
    ):
        """
        Initialize ScienceQA dataset.
        
        Args:
            data_dir: Directory containing the ScienceQA data files
            split: Data split to load ("train", "val", or "test")
            image_dir: Directory containing images (if None, will look for 'images' in data_dir)
            load_images: Whether to load images into memory
        """
        self.data_dir = data_dir
        self.split = split
        self.image_dir = image_dir or os.path.join(data_dir, "images")
        self.load_images = load_images
        
        # Load the dataset
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load the ScienceQA dataset.
        
        Returns:
            List of examples
        """
        try:
            # Load JSON data
            problems_path = os.path.join(self.data_dir, f"problems.json")
            pid_splits_path = os.path.join(self.data_dir, f"pid_splits.json")
            
            if not os.path.exists(problems_path) or not os.path.exists(pid_splits_path):
                raise FileNotFoundError(f"Required files missing in {self.data_dir}")
            
            with open(problems_path, "r") as f:
                problems = json.load(f)
                
            with open(pid_splits_path, "r") as f:
                pid_splits = json.load(f)
            
            # Get problem IDs for the specified split
            pids = pid_splits[self.split]
            
            # Process examples
            examples = []
            for pid in tqdm(pids, desc=f"Loading {self.split} data"):
                problem = problems[pid]
                
                example = {
                    "id": pid,
                    "question": problem["question"],
                    "options": problem.get("choices", []),
                    "answer": problem.get("answer"),
                    "subject": problem.get("subject", ""),
                    "topic": problem.get("topic", ""),
                    "context": problem.get("hint", ""),
                    "has_image": problem.get("has_image", False),
                    "image_path": None
                }
                
                # Process image if available
                if example["has_image"]:
                    img_folder = os.path.join(
                        self.image_dir, 
                        problem.get("subject", ""),
                        problem.get("topic", "")
                    )
                    
                    image_path = os.path.join(img_folder, f"{pid}.jpg")
                    if os.path.exists(image_path):
                        example["image_path"] = image_path
                        
                        if self.load_images:
                            try:
                                example["image"] = Image.open(image_path).convert("RGB")
                            except Exception as e:
                                logger.warning(f"Error loading image {image_path}: {e}")
                                example["image"] = None
                
                examples.append(example)
            
            logger.info(f"Loaded {len(examples)} examples from {self.split} split")
            return examples
            
        except Exception as e:
            logger.error(f"Error loading ScienceQA dataset: {e}")
            return []
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an example by index."""
        return self.data[idx]
    
    def get_example_by_id(self, example_id: str) -> Optional[Dict[str, Any]]:
        """Get an example by ID."""
        for example in self.data:
            if example["id"] == example_id:
                return example
        return None
    
    def get_multimodal_subset(self) -> List[Dict[str, Any]]:
        """Get only examples with images."""
        return [example for example in self.data if example["has_image"]]
    
    def get_random_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Get a random batch of examples."""
        if batch_size > len(self.data):
            batch_size = len(self.data)
        return random.sample(self.data, batch_size)
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get the distribution of examples by subject."""
        subjects = {}
        for example in self.data:
            subject = example["subject"]
            if subject in subjects:
                subjects[subject] += 1
            else:
                subjects[subject] = 1
        return subjects
    
    def get_example_for_mcts(self, idx: int) -> Dict[str, Any]:
        """
        Format an example for use with MCTS.
        
        Args:
            idx: Index of the example
            
        Returns:
            Formatted example
        """
        example = self.data[idx]
        
        mcts_example = {
            "question": example["question"],
            "options": example["options"],
            "context": example["context"],
            "image": example.get("image"),
            "image_path": example.get("image_path"),
            "has_image": example["has_image"],
            "id": example["id"],
            "image_description": "",
            "extracted_text": "",
            "history": [],
            "current_answer": "",
            "final_answer": None
        }
        
        return mcts_example 