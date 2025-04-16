import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import random
from PIL import Image

class ScienceQADataset:
    """Dataset class for ScienceQA dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        image_dir: Optional[str] = None,
        load_images: bool = False
    ):
        """
        Initialize the ScienceQA dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split (train, val, test)
            image_dir: Directory containing images (optional)
            load_images: Whether to load images
        """
        self.data_dir = data_dir
        self.split = split
        self.image_dir = image_dir
        self.load_images = load_images
        self.logger = logging.getLogger(__name__)
        
        # Define paths
        self.problems_path = os.path.join(data_dir, "scienceqa", "problems.json")
        self.pid_splits_path = os.path.join(data_dir, "scienceqa", "pid_splits.json")
        self.captions_path = os.path.join(data_dir, "captions.json")
        
        # Check if files exist
        if not os.path.exists(self.problems_path):
            alt_problems_path = os.path.join(data_dir, "ScienceQA-main", "data", "scienceqa", "problems.json")
            if os.path.exists(alt_problems_path):
                self.problems_path = alt_problems_path
                self.logger.info(f"Using alternative problems.json path: {alt_problems_path}")
            else:
                # Fall back to test_data.json if real dataset not available
                self.logger.warning(f"Problems file not found. Using test data instead.")
                self.problems_path = os.path.join(data_dir, "test_data.json")
                self.pid_splits_path = None
                self.captions_path = None
                self._load_test_data()
                return
                
        if not os.path.exists(self.pid_splits_path):
            alt_pid_splits_path = os.path.join(data_dir, "ScienceQA-main", "data", "scienceqa", "pid_splits.json")
            if os.path.exists(alt_pid_splits_path):
                self.pid_splits_path = alt_pid_splits_path
                self.logger.info(f"Using alternative pid_splits.json path: {alt_pid_splits_path}")
                
        if not os.path.exists(self.captions_path):
            alt_captions_path = os.path.join(data_dir, "ScienceQA-main", "data", "captions.json")
            if os.path.exists(alt_captions_path):
                self.captions_path = alt_captions_path
                self.logger.info(f"Using alternative captions.json path: {alt_captions_path}")
                
        # Load data
        self._load_data()
        
    def _load_test_data(self):
        """Load test data when the real dataset is not available."""
        test_data_path = os.path.join(self.data_dir, "test_data.json")
        if not os.path.exists(test_data_path):
            self.logger.warning(f"Test data file {test_data_path} not found")
            self.data = []
            self.problem_ids = []
            return
            
        try:
            with open(test_data_path, "r") as f:
                self.data = json.load(f)
                self.problem_ids = [item["id"] for item in self.data]
                
            self.logger.info(f"Loaded {len(self.data)} examples from {test_data_path}")
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            self.data = []
            self.problem_ids = []
    
    def _load_data(self):
        """Load data from files."""
        try:
            # Load problems
            with open(self.problems_path, "r") as f:
                self.problems = json.load(f)
                
            # Load split IDs
            if self.pid_splits_path and os.path.exists(self.pid_splits_path):
                with open(self.pid_splits_path, "r") as f:
                    self.pid_splits = json.load(f)
                self.problem_ids = self.pid_splits.get(self.split, [])
            else:
                self.pid_splits = {}
                self.problem_ids = list(self.problems.keys())
                
            # Load captions if available
            self.captions = {}
            if self.captions_path and os.path.exists(self.captions_path):
                try:
                    with open(self.captions_path, "r") as f:
                        self.captions = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Error loading captions: {e}")
            
            self.logger.info(f"Loaded {len(self.problem_ids)} examples for split {self.split}")
            
            # Create formatted data
            self.data = []
            for pid in self.problem_ids:
                problem = self.problems.get(pid, {})
                if problem:
                    formatted_problem = self._format_problem(pid, problem)
                    self.data.append(formatted_problem)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.problems = {}
            self.pid_splits = {}
            self.problem_ids = []
            self.captions = {}
            self.data = []
    
    def _format_problem(self, pid: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Format a problem for easier access."""
        choices = problem.get("choices", [])
        answer_idx = problem.get("answer", 0)
        answer = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
        
        formatted = {
            "id": pid,
            "question": problem.get("question", ""),
            "choices": choices,
            "answer": answer,
            "answer_idx": answer_idx,
            "subject": problem.get("subject", ""),
            "topic": problem.get("topic", ""),
            "category": problem.get("category", ""),
            "grade": problem.get("grade", ""),
            "image_file": problem.get("image", ""),
            "has_image": bool(problem.get("image", "")),
            "hint": problem.get("hint", ""),
            "lecture": problem.get("lecture", ""),
            "solution": problem.get("solution", ""),
            "caption": self.captions.get(pid, "")
        }
        
        return formatted
    
    def get_image(self, example_id: str) -> Optional[Image.Image]:
        """Get the image for an example if available."""
        if not self.image_dir:
            return None
            
        problem = self.problems.get(example_id, {})
        image_file = problem.get("image", "")
        
        if not image_file:
            return None
            
        # Construct image path
        image_path = os.path.join(self.image_dir, example_id, image_file)
        
        if not os.path.exists(image_path):
            self.logger.warning(f"Image file not found: {image_path}")
            return None
            
        try:
            return Image.open(image_path)
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an example by index."""
        if 0 <= idx < len(self.data):
            example = self.data[idx]
            
            # Load image if requested
            if self.load_images and example.get("has_image", False):
                image = self.get_image(example["id"])
                if image:
                    example["image"] = image
                    
            return example
        else:
            self.logger.warning(f"Index {idx} out of range")
            return {}
    
    def get_by_id(self, example_id: str) -> Dict[str, Any]:
        """Get an example by ID."""
        for example in self.data:
            if example["id"] == example_id:
                return example
        return {}
    
    def get_multimodal_subset(self) -> List[Dict[str, Any]]:
        """Get the subset of examples that require multimodal reasoning."""
        return [example for example in self.data if example.get("has_image", False)]
    
    def get_example_for_mcts(self, idx: Union[int, str]) -> Dict[str, Any]:
        """Format example for MCTS."""
        # Handle getting by ID or index
        if isinstance(idx, str):
            example = self.get_by_id(idx)
        elif isinstance(idx, int) and 0 <= idx < len(self.data):
            example = self.data[idx]
        else:
            self.logger.warning(f"Invalid index or ID: {idx}")
            return {}
        
        # Format for MCTS
        mcts_example = {
            "id": example.get("id", ""),
            "question": example.get("question", ""),
            "choices": example.get("choices", []),
            "answer": example.get("answer", ""),
            "subject": example.get("subject", ""),
            "topic": example.get("topic", ""),
            "category": example.get("category", ""),
            "has_image": example.get("has_image", False),
            "caption": example.get("caption", ""),
            "hint": example.get("hint", ""),
            "lecture": example.get("lecture", ""),
        }
        
        # Load image if available
        if example.get("has_image", False) and self.image_dir:
            image = self.get_image(example["id"])
            if image:
                mcts_example["image"] = image
        
        return mcts_example 