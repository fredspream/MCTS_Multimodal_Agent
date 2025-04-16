import os
import logging
import argparse
from typing import Dict, Any
import json
from PIL import Image

from src.multimodal_mcts_qa import MultimodalMCTSQA
from src.datasets import ScienceQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MultimodalMCTS for Question Answering')
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/scienceqa',
        help='Path to the ScienceQA dataset'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=5,
        help='Number of examples to evaluate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run models on'
    )
    
    parser.add_argument(
        '--llava_model',
        type=str,
        default='llava-hf/llava-1.5-7b-hf',
        help='LLaVA model name or path'
    )
    
    parser.add_argument(
        '--llama_model',
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
        help='LLaMA model name or path'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=50,
        help='Maximum number of MCTS iterations'
    )
    
    parser.add_argument(
        '--time_limit',
        type=float,
        default=30.0,
        help='Time limit for MCTS search in seconds'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset
    logger.info(f"Loading ScienceQA dataset from {args.dataset_path}")
    dataset = ScienceQADataset(args.dataset_path, split=args.split)
    
    # Initialize MCTS QA system
    logger.info(f"Initializing MultimodalMCTS QA system")
    mcts_qa = MultimodalMCTSQA(
        llava_model_name=args.llava_model,
        llama_model_name=args.llama_model,
        device=args.device,
        mcts_config={
            "max_iterations": args.max_iterations,
            "time_limit": args.time_limit,
            "exploration_weight": 1.0
        }
    )
    
    # Evaluate on dataset
    logger.info(f"Evaluating on {args.num_examples} examples from {args.split} split")
    results = mcts_qa.evaluate_on_dataset(
        dataset,
        num_examples=args.num_examples,
        output_dir=args.output_dir
    )
    
    # Print summary results
    logger.info(f"Evaluation complete")
    logger.info(f"Accuracy: {results['accuracy']:.2f}")
    logger.info(f"Correct: {results['correct']}/{results['num_examples']}")
    
    # Save summary results
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

def interactive_demo():
    """Run an interactive demo with user-provided questions and images."""
    logger.info("Starting interactive demo")
    
    # Initialize MCTS QA system
    logger.info("Initializing MultimodalMCTS QA system")
    mcts_qa = MultimodalMCTSQA()
    
    while True:
        # Get question from user
        question = input("\nEnter your question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
        
        # Get image path from user
        image_path = input("Enter image path (or press Enter to skip): ")
        
        # Create example
        example = {"question": question}
        
        # Load image if provided
        if image_path and os.path.exists(image_path):
            example["image"] = image_path
        
        # Answer the question
        result = mcts_qa.answer_question(example)
        
        # Print the result
        print("\n===== Result =====")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        print("\nReasoning path:")
        for step in result['reasoning_path']:
            print(f"  - {step['action']}: {step['observation']}")
        
        print("\nTools used:")
        for tool in result['tools_used']:
            print(f"  - {tool}")

if __name__ == "__main__":
    main() 