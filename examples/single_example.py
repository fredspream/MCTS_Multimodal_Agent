import os
import sys
import json
import logging
from PIL import Image
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MultimodalMCTSQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Example of using the Multimodal MCTS QA system with a single example.
    """
    # Initialize the system
    logger.info("Initializing Multimodal MCTS QA system")
    system = MultimodalMCTSQA(
        # Use smaller models for demonstration
        llava_model_name="llava-hf/llava-1.5-7b-hf",
        llama_model_name="meta-llama/Llama-2-7b-chat-hf",
        # Explicitly set to CPU for this example to avoid CUDA errors
        device="cpu",
        # Shorter search time for demonstration
        mcts_config={
            "time_limit": 10.0,
            "max_iterations": 20,
            "exploration_weight": 1.0
        }
    )
    
    # Example image path - replace with your own image
    image_path = "path/to/your/example_image.jpg"
    
    # Ensure the image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        logger.info("Please replace with a valid image path")
        return
    
    # Create an example
    example = {
        "question": "What is shown in this image?",
        "options": ["A cat", "A dog", "A bird", "A fish"],
        "context": "",
        "image": Image.open(image_path).convert("RGB"),
        "image_path": image_path,
        "has_image": True,
        "id": "example_1",
        "image_description": "",
        "extracted_text": "",
        "history": [],
        "current_answer": "",
        "final_answer": None
    }
    
    # Process the example
    logger.info(f"Processing example with question: {example['question']}")
    result = system.answer_question(example)
    
    # Print results
    logger.info("Result:")
    logger.info(f"Question: {result['question']}")
    logger.info(f"Answer: {result['final_answer']}")
    logger.info(f"Image description: {result['image_description']}")
    
    if result['extracted_text']:
        logger.info(f"Extracted text: {result['extracted_text']}")
    
    logger.info("Reasoning path:")
    for step in result['reasoning_path']:
        logger.info(f"- Action: {step['action']}")
        logger.info(f"  Input: {step['action_input']}")
        logger.info(f"  Observation: {step['observation']}")
    
    # Save results
    os.makedirs("example_results", exist_ok=True)
    with open("example_results/single_example_result.json", "w") as f:
        # Convert PIL image to path for JSON serialization
        result_for_json = result.copy()
        if 'image' in result_for_json:
            del result_for_json['image']
        json.dump(result_for_json, f, indent=2)
    
    logger.info("Results saved to example_results/single_example_result.json")

if __name__ == "__main__":
    main() 