#!/usr/bin/env python
"""
Evaluate MCTS tool-use strategies on the ScienceQA test set.
This script runs evaluation and analyzes how MCTS discovers and utilizes different tools.
"""

import os
import argparse
import logging
import json
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import torch
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MultimodalMCTSQA
from src.data import ScienceQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MCTS Tool-Use Strategies")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=50,
        help="Number of examples to evaluate"
    )
    
    parser.add_argument(
        "--record_search_traces", 
        action="store_true",
        help="Record and analyze full MCTS search traces"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def analyze_tool_usage(results):
    """
    Analyze tool usage patterns in the results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary of analysis metrics
    """
    # Initialize counters
    total_queries = len(results)
    tool_counts = Counter()
    subject_tool_usage = defaultdict(lambda: defaultdict(int))
    tool_sequences = []
    
    # Analyze each query result
    for result in results:
        # Extract tool sequence
        tools_used = [step["action"] for step in result.get("reasoning_path", [])]
        tool_sequences.append(tools_used)
        
        # Count tools used
        for tool in tools_used:
            tool_counts[tool] += 1
            
        # Record subject-specific tool usage
        subject = result.get("subject", "unknown")
        for tool in set(tools_used):  # Count each tool only once per query
            subject_tool_usage[subject][tool] += 1
    
    # Calculate average tools used per query
    avg_tools_per_query = sum(len(seq) for seq in tool_sequences) / total_queries
    
    # Calculate most common tool sequences
    sequence_counter = Counter(tuple(seq) for seq in tool_sequences)
    most_common_sequences = sequence_counter.most_common(5)
    
    # Calculate tool transition probabilities
    transitions = defaultdict(Counter)
    for seq in tool_sequences:
        for i in range(len(seq) - 1):
            transitions[seq[i]][seq[i + 1]] += 1
    
    # Normalize transitions to probabilities
    transition_probs = {}
    for tool1, next_tools in transitions.items():
        total = sum(next_tools.values())
        transition_probs[tool1] = {tool2: count / total for tool2, count in next_tools.items()}
    
    return {
        "total_queries": total_queries,
        "tool_counts": dict(tool_counts),
        "avg_tools_per_query": avg_tools_per_query,
        "most_common_sequences": most_common_sequences,
        "transition_probs": transition_probs,
        "subject_tool_usage": {k: dict(v) for k, v in subject_tool_usage.items()}
    }

def visualize_tool_usage(analysis, output_dir):
    """
    Create visualizations of tool usage patterns.
    
    Args:
        analysis: Analysis dictionary from analyze_tool_usage
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tool usage counts
    plt.figure(figsize=(10, 6))
    tools = list(analysis["tool_counts"].keys())
    counts = list(analysis["tool_counts"].values())
    plt.bar(tools, counts)
    plt.title("Tool Usage Counts")
    plt.xlabel("Tool")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "tool_counts.png"))
    plt.close()
    
    # 2. Tool usage by subject
    subject_data = analysis["subject_tool_usage"]
    if len(subject_data) > 1:  # Only if we have multiple subjects
        plt.figure(figsize=(12, 8))
        subjects = list(subject_data.keys())
        tools = set()
        for subject_tools in subject_data.values():
            tools.update(subject_tools.keys())
        tools = sorted(tools)
        
        bar_width = 0.8 / len(subjects)
        index = np.arange(len(tools))
        
        for i, subject in enumerate(subjects):
            subject_counts = [subject_data[subject].get(tool, 0) for tool in tools]
            plt.bar(index + i * bar_width, subject_counts, bar_width, label=subject)
        
        plt.xlabel('Tools')
        plt.ylabel('Usage Count')
        plt.title('Tool Usage by Subject')
        plt.xticks(index + bar_width * (len(subjects) - 1) / 2, tools)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tool_usage_by_subject.png"))
        plt.close()
    
    # 3. Transition probabilities (as a heat map)
    transition_probs = analysis["transition_probs"]
    if transition_probs:
        all_tools = set()
        for tool1, next_tools in transition_probs.items():
            all_tools.add(tool1)
            all_tools.update(next_tools.keys())
        all_tools = sorted(all_tools)
        
        matrix = np.zeros((len(all_tools), len(all_tools)))
        for i, tool1 in enumerate(all_tools):
            if tool1 in transition_probs:
                for j, tool2 in enumerate(all_tools):
                    matrix[i, j] = transition_probs[tool1].get(tool2, 0)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar(label='Transition Probability')
        plt.title('Tool Transition Probabilities')
        plt.xlabel('Next Tool')
        plt.ylabel('Current Tool')
        plt.xticks(range(len(all_tools)), all_tools, rotation=45)
        plt.yticks(range(len(all_tools)), all_tools)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "transition_probabilities.png"))
        plt.close()

def analyze_search_tree(mcts_system, example, output_dir):
    """
    Analyze and visualize the MCTS search tree for a single example.
    
    Args:
        mcts_system: MultimodalMCTSQA system
        example: Example to analyze
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary with tree analysis metrics
    """
    # Run MCTS search
    mcts = mcts_system._setup_mcts()
    mcts_system.current_state = example
    
    # Track statistics during search
    tree_stats = {
        "iterations": 0,
        "nodes_created": 0,
        "max_depth": 0,
        "branching_factor": [],
        "node_visits": defaultdict(int),
        "reward_distribution": []
    }
    
    # Run MCTS iterations with tracking
    root = mcts._run_mcts_with_tracking(example, tree_stats)
    
    # Save tree stats
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tree_stats.json"), "w") as f:
        # Convert defaultdict to regular dict for JSON serialization
        tree_stats["node_visits"] = dict(tree_stats["node_visits"])
        json.dump(tree_stats, f, indent=2)
    
    # Visualize reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(tree_stats["reward_distribution"], bins=20)
    plt.title("Reward Distribution in MCTS Search")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
    plt.close()
    
    # Visualize node visit distribution
    plt.figure(figsize=(10, 6))
    visits = list(tree_stats["node_visits"].values())
    plt.hist(visits, bins=range(1, max(visits) + 2))
    plt.title("Node Visit Distribution")
    plt.xlabel("Number of Visits")
    plt.ylabel("Number of Nodes")
    plt.savefig(os.path.join(output_dir, "node_visits.png"))
    plt.close()
    
    return tree_stats

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load ScienceQA dataset
    logger.info(f"Loading ScienceQA test set from {config['dataset']['data_dir']}")
    dataset = ScienceQADataset(
        data_dir=config["dataset"]["data_dir"],
        split="test",  # Use test split explicitly
        image_dir=config["dataset"]["image_dir"],
        load_images=config["dataset"]["load_images"]
    )
    
    # Initialize OCR configuration
    ocr_config = config["ocr"]
    
    # Initialize the Multimodal MCTS QA system
    logger.info("Initializing Multimodal MCTS QA system")
    system = MultimodalMCTSQA(
        llava_model_name=config["models"]["llava"]["model_name"],
        llama_model_name=config["models"]["llama"]["model_name"],
        device=config["models"]["llava"]["device"],
        ocr_config=ocr_config,
        mcts_config=config["mcts"]
    )
    
    # Patch the MCTS class to track search tree statistics if needed
    if args.record_search_traces:
        # Add method to MultimodalMCTSQA class to set up MCTS
        def _setup_mcts(self):
            mcts = self.mcts_class(
                state_generator=self._state_generator,
                simulator=self._simulator,
                time_limit=self.mcts_config["time_limit"],
                max_iterations=self.mcts_config["max_iterations"],
                exploration_weight=self.mcts_config["exploration_weight"]
            )
            return mcts
        
        # Add method to MCTS class to run with tracking
        def _run_mcts_with_tracking(self, initial_state, stats):
            root = self._create_root_node(initial_state)
            
            start_time = self._get_current_time()
            iterations = 0
            
            # Track the number of nodes created
            initial_nodes = len(root.children)
            
            # Main MCTS loop
            while (self._get_current_time() - start_time < self.time_limit and 
                   iterations < self.max_iterations):
                
                # 1. Selection: traverse the tree to find a node to expand
                node = self._select(root)
                
                # Track node depth
                depth = 0
                temp_node = node
                while temp_node.parent is not None:
                    depth += 1
                    temp_node = temp_node.parent
                stats["max_depth"] = max(stats["max_depth"], depth)
                
                # 2. Expansion: add a new child to the selected node
                if not node.is_terminal() and not node.is_fully_expanded():
                    node = self._expand(node)
                
                # 3. Simulation: perform rollout from the selected node
                reward = self._simulate(node)
                stats["reward_distribution"].append(reward)
                
                # 4. Backpropagation: update statistics along the path
                self._backpropagate(node, reward)
                
                # Track node visits
                temp_node = node
                while temp_node is not None:
                    stats["node_visits"][id(temp_node)] += 1
                    temp_node = temp_node.parent
                
                iterations += 1
            
            stats["iterations"] = iterations
            stats["nodes_created"] = len(root.children) - initial_nodes
            
            if root.children:
                stats["branching_factor"] = [len(child.children) for child in root.children]
            
            return root
        
        # Monkey patch the methods
        from types import MethodType
        system._setup_mcts = MethodType(_setup_mcts, system)
        system.mcts_class = system.mcts.__class__
        system.mcts_class._run_mcts_with_tracking = _run_mcts_with_tracking
        system.mcts_class._create_root_node = lambda self, state: self.node_class(state=state, exploration_weight=self.exploration_weight)
        system.mcts_class._get_current_time = lambda self: time.time()
    
    # Get multimodal examples from test set
    multimodal_examples = dataset.get_multimodal_subset()
    logger.info(f"Found {len(multimodal_examples)} multimodal examples in the test set")
    
    # Limit the number of examples
    if args.num_examples > len(multimodal_examples):
        args.num_examples = len(multimodal_examples)
    test_examples = multimodal_examples[:args.num_examples]
    
    # Process each example
    logger.info(f"Evaluating on {len(test_examples)} examples")
    results = []
    
    for i, example in enumerate(tqdm(test_examples, desc="Processing examples")):
        logger.info(f"Processing example {i+1}/{len(test_examples)}")
        
        # Format example for MCTS
        mcts_example = dataset.get_example_for_mcts(
            dataset.data.index(example)
        )
        
        # Answer the question
        result = system.answer_question(mcts_example)
        
        # Add metadata
        result["subject"] = example.get("subject", "")
        result["topic"] = example.get("topic", "")
        result["ground_truth"] = example.get("answer", "")
        result["is_correct"] = result["final_answer"] == result["ground_truth"]
        
        # Analyze search tree if requested
        if args.record_search_traces and i < 5:  # Limit to first 5 examples to save time
            tree_stats_dir = os.path.join(args.output_dir, f"tree_stats_example_{i}")
            tree_stats = analyze_search_tree(system, mcts_example, tree_stats_dir)
            result["tree_stats_dir"] = tree_stats_dir
        
        # Add to results
        results.append(result)
    
    # Save all results
    with open(os.path.join(args.output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Analyze tool usage
    logger.info("Analyzing tool usage patterns")
    tool_analysis = analyze_tool_usage(results)
    
    # Save analysis
    with open(os.path.join(args.output_dir, "tool_usage_analysis.json"), "w") as f:
        json.dump(tool_analysis, f, indent=2)
    
    # Visualize tool usage
    logger.info("Creating visualizations")
    visualize_tool_usage(tool_analysis, args.output_dir)
    
    # Print summary
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / len(results) if results else 0
    
    logger.info(f"Evaluation completed")
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    logger.info(f"Average tools used per query: {tool_analysis['avg_tools_per_query']:.2f}")
    logger.info(f"Most common tool: {max(tool_analysis['tool_counts'].items(), key=lambda x: x[1])[0]}")
    
    # Save summary
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})\n")
        f.write(f"Average tools used per query: {tool_analysis['avg_tools_per_query']:.2f}\n")
        f.write(f"Tool usage counts: {tool_analysis['tool_counts']}\n")
        f.write(f"Most common sequences: {tool_analysis['most_common_sequences']}\n")

if __name__ == "__main__":
    import time
    main() 