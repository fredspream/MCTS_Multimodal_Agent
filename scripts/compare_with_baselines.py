#!/usr/bin/env python
"""
Compare the MultimodalMCTS approach against baseline methods on multimodal QA tasks.
Baseline methods include:
1. Direct LLM answering (no tool use)
2. Random tool selection
3. Fixed tool sequence based on question type
4. Human-designed heuristics
"""

import os
import argparse
import logging
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import importlib.util
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MultimodalMCTS with baseline methods")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of examples to evaluate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baseline_comparison_results",
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory with pre-computed results (if available)"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def import_mcts_system(config):
    """Import the MultimodalMCTS system based on config."""
    logger.info("Initializing MultimodalMCTS system...")
    
    try:
        # Get the module path from config
        mcts_module_path = config.get('mcts_module_path', 'src/multimodal_mcts.py')
        
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("multimodal_mcts", mcts_module_path)
        mcts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcts_module)
        
        # Get the system class
        system_class = getattr(mcts_module, config.get('mcts_class_name', 'MultimodalMCTS'))
        
        # Initialize the system
        system = system_class(**config.get('mcts_params', {}))
        
        logger.info("MultimodalMCTS system initialized successfully.")
        return system
    except Exception as e:
        logger.error(f"Failed to import MultimodalMCTS system: {e}")
        sys.exit(1)

def setup_baseline_methods(config, system):
    """Set up baseline methods for comparison."""
    logger.info("Setting up baseline methods...")
    
    baseline_methods = {
        "mcts": system,  # The full MultimodalMCTS system
    }
    
    try:
        # Direct LLM baseline (no tool use)
        llm_baseline_params = config.get('baseline_params', {}).get('llm', {})
        baseline_methods["llm_direct"] = DirectLLMBaseline(
            **llm_baseline_params,
            llm_model=system.llm_model if hasattr(system, 'llm_model') else None
        )
        
        # Random tool selection baseline
        random_baseline_params = config.get('baseline_params', {}).get('random', {})
        baseline_methods["random_tools"] = RandomToolBaseline(
            **random_baseline_params,
            available_tools=system.available_tools if hasattr(system, 'available_tools') else [],
            llm_model=system.llm_model if hasattr(system, 'llm_model') else None
        )
        
        # Fixed tool sequence baseline
        fixed_baseline_params = config.get('baseline_params', {}).get('fixed', {})
        baseline_methods["fixed_sequence"] = FixedSequenceBaseline(
            **fixed_baseline_params,
            available_tools=system.available_tools if hasattr(system, 'available_tools') else [],
            llm_model=system.llm_model if hasattr(system, 'llm_model') else None
        )
        
        # Heuristic-based baseline
        heuristic_baseline_params = config.get('baseline_params', {}).get('heuristic', {})
        baseline_methods["heuristic"] = HeuristicBaseline(
            **heuristic_baseline_params,
            available_tools=system.available_tools if hasattr(system, 'available_tools') else [],
            llm_model=system.llm_model if hasattr(system, 'llm_model') else None
        )
        
        logger.info(f"Set up {len(baseline_methods)} baseline methods successfully.")
        return baseline_methods
    except Exception as e:
        logger.error(f"Failed to set up baseline methods: {e}")
        sys.exit(1)

class DirectLLMBaseline:
    """Baseline using direct LLM answers without tool use."""
    
    def __init__(self, llm_model=None, max_tokens=100, temperature=0.0, **kwargs):
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = "Direct LLM (No Tools)"
    
    def answer_question(self, question, image=None, context=None):
        """Answer a question directly without using tools."""
        prompt = self._construct_prompt(question, context)
        
        try:
            if image is not None and hasattr(self.llm_model, 'generate_with_image'):
                response = self.llm_model.generate_with_image(
                    prompt, 
                    image, 
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            else:
                response = self.llm_model.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            
            return {
                'answer': response,
                'tools_used': [],
                'reasoning': "Direct answer without tools",
                'confidence': 1.0
            }
        except Exception as e:
            logger.error(f"Error in DirectLLMBaseline: {e}")
            return {
                'answer': "Failed to generate an answer",
                'tools_used': [],
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0
            }
    
    def _construct_prompt(self, question, context=None):
        """Construct a prompt for the LLM."""
        prompt = "Answer the following question directly and concisely:\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
            
        prompt += f"Question: {question}\n\nAnswer: "
        return prompt

class RandomToolBaseline:
    """Baseline that uses random tool selection."""
    
    def __init__(self, available_tools=None, llm_model=None, max_steps=5, **kwargs):
        self.available_tools = available_tools or []
        self.llm_model = llm_model
        self.max_steps = max_steps
        self.name = "Random Tool Selection"
    
    def answer_question(self, question, image=None, context=None):
        """Answer a question using randomly selected tools."""
        tools_used = []
        gathered_info = []
        
        # Randomly use tools up to max_steps
        steps = np.random.randint(0, self.max_steps + 1)
        
        for _ in range(steps):
            if not self.available_tools:
                break
                
            # Randomly select a tool
            tool_idx = np.random.randint(0, len(self.available_tools))
            tool = self.available_tools[tool_idx]
            
            try:
                # Use the tool
                if hasattr(tool, 'use'):
                    result = tool.use(question, image, context)
                    tools_used.append(tool.__class__.__name__)
                    gathered_info.append(f"{tool.__class__.__name__}: {result}")
            except Exception as e:
                logger.warning(f"Error using tool {tool.__class__.__name__}: {e}")
        
        # Construct answer using gathered information
        prompt = self._construct_final_prompt(question, gathered_info, context)
        
        try:
            if image is not None and hasattr(self.llm_model, 'generate_with_image'):
                answer = self.llm_model.generate_with_image(prompt, image)
            else:
                answer = self.llm_model.generate(prompt)
                
            return {
                'answer': answer,
                'tools_used': tools_used,
                'reasoning': "Random tool selection",
                'confidence': 0.5
            }
        except Exception as e:
            logger.error(f"Error in RandomToolBaseline: {e}")
            return {
                'answer': "Failed to generate an answer",
                'tools_used': tools_used,
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0
            }
    
    def _construct_final_prompt(self, question, gathered_info, context=None):
        """Construct the final prompt for answering the question."""
        prompt = "Based on the following information, answer the question concisely:\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        if gathered_info:
            prompt += "Information gathered:\n"
            for info in gathered_info:
                prompt += f"- {info}\n"
            prompt += "\n"
            
        prompt += f"Question: {question}\n\nAnswer: "
        return prompt

class FixedSequenceBaseline:
    """Baseline that uses fixed tool sequences based on question type."""
    
    def __init__(self, available_tools=None, llm_model=None, sequence_map=None, **kwargs):
        self.available_tools = {tool.__class__.__name__: tool for tool in (available_tools or [])}
        self.llm_model = llm_model
        self.sequence_map = sequence_map or {
            "image": ["ImageCaptionTool", "ObjectDetectionTool"],
            "text": ["TextExtractionTool", "KeywordExtractionTool"],
            "calculation": ["CalculatorTool"],
            "knowledge": ["KnowledgeRetrievalTool"],
            "default": ["ImageCaptionTool", "TextExtractionTool"]
        }
        self.name = "Fixed Tool Sequence"
    
    def answer_question(self, question, image=None, context=None):
        """Answer a question using fixed tool sequences based on question type."""
        # Determine question type
        question_type = self._determine_question_type(question)
        
        # Get the tool sequence for this question type
        tool_sequence = self.sequence_map.get(question_type, self.sequence_map["default"])
        
        tools_used = []
        gathered_info = []
        
        # Use tools in sequence
        for tool_name in tool_sequence:
            if tool_name in self.available_tools:
                tool = self.available_tools[tool_name]
                
                try:
                    # Use the tool
                    if hasattr(tool, 'use'):
                        result = tool.use(question, image, context)
                        tools_used.append(tool_name)
                        gathered_info.append(f"{tool_name}: {result}")
                except Exception as e:
                    logger.warning(f"Error using tool {tool_name}: {e}")
        
        # Construct answer using gathered information
        prompt = self._construct_final_prompt(question, gathered_info, context)
        
        try:
            if image is not None and hasattr(self.llm_model, 'generate_with_image'):
                answer = self.llm_model.generate_with_image(prompt, image)
            else:
                answer = self.llm_model.generate(prompt)
                
            return {
                'answer': answer,
                'tools_used': tools_used,
                'reasoning': f"Fixed sequence for {question_type} questions",
                'confidence': 0.7
            }
        except Exception as e:
            logger.error(f"Error in FixedSequenceBaseline: {e}")
            return {
                'answer': "Failed to generate an answer",
                'tools_used': tools_used,
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0
            }
    
    def _determine_question_type(self, question):
        """Determine the type of question based on keywords."""
        question = question.lower()
        
        if any(kw in question for kw in ["image", "picture", "photo", "color", "object", "show"]):
            return "image"
        elif any(kw in question for kw in ["text", "write", "read", "say"]):
            return "text"
        elif any(kw in question for kw in ["calculate", "how many", "sum", "difference", "product"]):
            return "calculation"
        elif any(kw in question for kw in ["who", "when", "where", "why", "what is", "explain"]):
            return "knowledge"
        else:
            return "default"
    
    def _construct_final_prompt(self, question, gathered_info, context=None):
        """Construct the final prompt for answering the question."""
        prompt = "Based on the following information, answer the question concisely:\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        if gathered_info:
            prompt += "Information gathered:\n"
            for info in gathered_info:
                prompt += f"- {info}\n"
            prompt += "\n"
            
        prompt += f"Question: {question}\n\nAnswer: "
        return prompt

class HeuristicBaseline:
    """Baseline using heuristics to determine tool usage."""
    
    def __init__(self, available_tools=None, llm_model=None, **kwargs):
        self.available_tools = {tool.__class__.__name__: tool for tool in (available_tools or [])}
        self.llm_model = llm_model
        self.name = "Heuristic-based"
    
    def answer_question(self, question, image=None, context=None):
        """Answer a question using heuristic-based tool selection."""
        # First, analyze the question to decide which tools to use
        tools_to_use = self._select_tools(question, image is not None)
        
        tools_used = []
        gathered_info = []
        
        # Use selected tools
        for tool_name in tools_to_use:
            if tool_name in self.available_tools:
                tool = self.available_tools[tool_name]
                
                try:
                    # Use the tool
                    if hasattr(tool, 'use'):
                        result = tool.use(question, image, context)
                        tools_used.append(tool_name)
                        gathered_info.append(f"{tool_name}: {result}")
                except Exception as e:
                    logger.warning(f"Error using tool {tool_name}: {e}")
        
        # Construct answer using gathered information
        prompt = self._construct_final_prompt(question, gathered_info, context)
        
        try:
            if image is not None and hasattr(self.llm_model, 'generate_with_image'):
                answer = self.llm_model.generate_with_image(prompt, image)
            else:
                answer = self.llm_model.generate(prompt)
                
            return {
                'answer': answer,
                'tools_used': tools_used,
                'reasoning': "Heuristic-based tool selection",
                'confidence': 0.8
            }
        except Exception as e:
            logger.error(f"Error in HeuristicBaseline: {e}")
            return {
                'answer': "Failed to generate an answer",
                'tools_used': tools_used,
                'reasoning': f"Error: {str(e)}",
                'confidence': 0.0
            }
    
    def _select_tools(self, question, has_image):
        """Select tools based on question analysis."""
        question = question.lower()
        selected_tools = []
        
        # Image analysis tools
        if has_image:
            if any(kw in question for kw in ["color", "object", "show", "appear", "look"]):
                selected_tools.append("ObjectDetectionTool")
            
            if "caption" in question or "describe" in question or len(question.split()) < 10:
                selected_tools.append("ImageCaptionTool")
                
            if any(kw in question for kw in ["text", "write", "read", "say"]):
                selected_tools.append("TextExtractionTool")
        
        # Text analysis tools
        if "keyword" in question or "topic" in question or "about" in question:
            selected_tools.append("KeywordExtractionTool")
            
        # Calculation tools
        if any(kw in question for kw in ["calculate", "how many", "sum", "difference", "product"]):
            selected_tools.append("CalculatorTool")
            
        # Knowledge tools
        if any(kw in question for kw in ["who", "when", "where", "why", "what is", "explain"]):
            selected_tools.append("KnowledgeRetrievalTool")
            
        # If no tools were selected, use direct answering
        if not selected_tools and has_image:
            selected_tools.append("ImageCaptionTool")
            
        return selected_tools
    
    def _construct_final_prompt(self, question, gathered_info, context=None):
        """Construct the final prompt for answering the question."""
        prompt = "Based on the following information, answer the question concisely:\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        if gathered_info:
            prompt += "Information gathered:\n"
            for info in gathered_info:
                prompt += f"- {info}\n"
            prompt += "\n"
            
        prompt += f"Question: {question}\n\nAnswer: "
        return prompt

def load_dataset(config):
    """Load the dataset for evaluation."""
    logger.info("Loading dataset...")
    
    dataset_path = config.get('dataset_path', 'data/scienceqa_test.json')
    
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)

def evaluate_methods(baseline_methods, dataset, num_examples, config, output_dir):
    """Evaluate all baseline methods on the dataset."""
    logger.info(f"Evaluating {len(baseline_methods)} methods on {num_examples} examples...")
    
    # Limit the number of examples
    examples = dataset[:num_examples] if num_examples > 0 else dataset
    
    results = {}
    
    for method_name, method in baseline_methods.items():
        logger.info(f"Evaluating method: {method_name}")
        method_results = []
        
        for example in tqdm(examples, desc=f"Evaluating {method_name}"):
            try:
                # Get question and image (if available)
                question = example.get('question', '')
                image_path = example.get('image_path')
                image = None
                
                if image_path and os.path.exists(image_path):
                    from PIL import Image
                    image = Image.open(image_path)
                
                # Answer the question
                result = method.answer_question(
                    question=question,
                    image=image,
                    context=example.get('context')
                )
                
                # Add ground truth and metadata
                result['question'] = question
                result['ground_truth'] = example.get('answer', '')
                result['id'] = example.get('id', '')
                result['domain'] = example.get('domain', '')
                result['category'] = example.get('category', '')
                
                # Add to results
                method_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating example with {method_name}: {e}")
                
        # Save method results
        results[method_name] = method_results
        
        # Save to file
        method_output_path = os.path.join(output_dir, f"{method_name}_results.json")
        with open(method_output_path, 'w') as f:
            json.dump(method_results, f, indent=2)
            
        logger.info(f"Saved results for {method_name} to {method_output_path}")
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics for each method."""
    logger.info("Calculating performance metrics...")
    
    metrics = {}
    
    for method_name, method_results in results.items():
        # Extract predictions and ground truth
        predictions = []
        ground_truth = []
        domains = []
        categories = []
        
        for result in method_results:
            if 'answer' in result and 'ground_truth' in result:
                predictions.append(result['answer'])
                ground_truth.append(result['ground_truth'])
                domains.append(result.get('domain', 'unknown'))
                categories.append(result.get('category', 'unknown'))
        
        # Calculate overall accuracy 
        # Note: In a real system, would need more sophisticated answer matching
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p.lower().strip() == g.lower().strip())
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate accuracy by domain
        domain_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
        for p, g, d in zip(predictions, ground_truth, domains):
            domain_acc[d]['total'] += 1
            if p.lower().strip() == g.lower().strip():
                domain_acc[d]['correct'] += 1
        
        domain_accuracy = {d: data['correct'] / data['total'] if data['total'] > 0 else 0 
                          for d, data in domain_acc.items()}
        
        # Calculate accuracy by category
        category_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
        for p, g, c in zip(predictions, ground_truth, categories):
            category_acc[c]['total'] += 1
            if p.lower().strip() == g.lower().strip():
                category_acc[c]['correct'] += 1
        
        category_accuracy = {c: data['correct'] / data['total'] if data['total'] > 0 else 0 
                            for c, data in category_acc.items()}
        
        # Tool usage analysis
        tool_usage = defaultdict(int)
        for result in method_results:
            for tool in result.get('tools_used', []):
                tool_usage[tool] += 1
        
        avg_tools_used = sum(len(result.get('tools_used', [])) for result in method_results) / len(method_results)
        
        # Store metrics
        metrics[method_name] = {
            'accuracy': accuracy,
            'total_examples': total,
            'domain_accuracy': domain_accuracy,
            'category_accuracy': category_accuracy,
            'tool_usage': dict(tool_usage),
            'avg_tools_used': avg_tools_used
        }
    
    return metrics

def generate_comparison_visualizations(metrics, output_dir, baseline_methods):
    """Generate visualizations comparing the methods."""
    logger.info("Generating comparison visualizations...")
    
    # Create accuracy comparison chart
    method_names = list(metrics.keys())
    accuracies = [metrics[name]['accuracy'] for name in method_names]
    
    # Rename method names for display
    display_names = [baseline_methods[name].name if name in baseline_methods else name for name in method_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(display_names, accuracies, color=sns.color_palette("viridis", len(method_names)))
    
    # Add values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom')
    
    plt.title('Accuracy Comparison Across Methods')
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    accuracy_chart_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(accuracy_chart_path)
    plt.close()
    
    # Create tool usage heatmap
    all_tools = set()
    for method_metrics in metrics.values():
        all_tools.update(method_metrics['tool_usage'].keys())
    
    all_tools = sorted(list(all_tools))
    
    # Create matrix of tool usage
    tool_usage_matrix = np.zeros((len(method_names), len(all_tools)))
    
    for i, method_name in enumerate(method_names):
        method_tool_usage = metrics[method_name]['tool_usage']
        for j, tool in enumerate(all_tools):
            tool_usage_matrix[i, j] = method_tool_usage.get(tool, 0)
    
    # Normalize by number of examples
    for i, method_name in enumerate(method_names):
        total_examples = metrics[method_name]['total_examples']
        if total_examples > 0:
            tool_usage_matrix[i, :] = tool_usage_matrix[i, :] / total_examples
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(tool_usage_matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=all_tools, yticklabels=display_names)
    plt.title('Tool Usage Comparison (Normalized by Example Count)')
    plt.xlabel('Tool')
    plt.ylabel('Method')
    plt.tight_layout()
    
    tool_usage_chart_path = os.path.join(output_dir, 'tool_usage_comparison.png')
    plt.savefig(tool_usage_chart_path)
    plt.close()
    
    # Create average tool usage chart
    avg_tools = [metrics[name]['avg_tools_used'] for name in method_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(display_names, avg_tools, color=sns.color_palette("viridis", len(method_names)))
    
    # Add values on top of bars
    for bar, val in zip(bars, avg_tools):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.title('Average Number of Tools Used')
    plt.xlabel('Method')
    plt.ylabel('Average Tools Used')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    avg_tools_chart_path = os.path.join(output_dir, 'avg_tools_used.png')
    plt.savefig(avg_tools_chart_path)
    plt.close()
    
    return {
        'accuracy_chart': accuracy_chart_path,
        'tool_usage_chart': tool_usage_chart_path,
        'avg_tools_chart': avg_tools_chart_path
    }

def generate_comparison_summary(metrics, visualization_paths, output_dir):
    """Generate a text summary of the comparison results."""
    logger.info("Generating comparison summary...")
    
    summary_path = os.path.join(output_dir, 'comparison_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("# Comparison with Baseline Methods\n\n")
        
        # Overall accuracy comparison
        f.write("## Overall Accuracy\n\n")
        method_names = list(metrics.keys())
        for method_name in method_names:
            acc = metrics[method_name]['accuracy']
            f.write(f"- {method_name}: {acc:.4f}\n")
        f.write("\n")
        
        # Best performing method
        best_method = max(method_names, key=lambda x: metrics[x]['accuracy'])
        best_acc = metrics[best_method]['accuracy']
        f.write(f"The best performing method is **{best_method}** with an accuracy of {best_acc:.4f}.\n\n")
        
        # Performance improvement
        if 'mcts' in metrics:
            mcts_acc = metrics['mcts']['accuracy']
            other_methods = [m for m in method_names if m != 'mcts']
            
            if other_methods:
                best_baseline = max(other_methods, key=lambda x: metrics[x]['accuracy'])
                best_baseline_acc = metrics[best_baseline]['accuracy']
                
                improvement = (mcts_acc - best_baseline_acc) / best_baseline_acc * 100
                f.write(f"MultimodalMCTS improves over the best baseline ({best_baseline}) by {improvement:.2f}%.\n\n")
        
        # Tool usage analysis
        f.write("## Tool Usage Analysis\n\n")
        
        if 'mcts' in metrics:
            mcts_tools = metrics['mcts']['tool_usage']
            most_used_tool = max(mcts_tools.items(), key=lambda x: x[1])[0] if mcts_tools else "None"
            
            f.write(f"The MultimodalMCTS approach most frequently uses the **{most_used_tool}** tool.\n")
            f.write(f"On average, it uses {metrics['mcts']['avg_tools_used']:.2f} tools per question.\n\n")
            
            f.write("Tool usage frequencies:\n")
            for tool, count in sorted(mcts_tools.items(), key=lambda x: x[1], reverse=True):
                percentage = count / metrics['mcts']['total_examples'] * 100
                f.write(f"- {tool}: {percentage:.1f}%\n")
            f.write("\n")
        
        # Domain analysis
        f.write("## Domain Analysis\n\n")
        
        all_domains = set()
        for method_metrics in metrics.values():
            all_domains.update(method_metrics['domain_accuracy'].keys())
        
        f.write("| Domain | " + " | ".join(method_names) + " |\n")
        f.write("|--------|" + "|".join(["-" * len(name) for name in method_names]) + "|\n")
        
        for domain in sorted(all_domains):
            f.write(f"| {domain} | ")
            for method in method_names:
                acc = metrics[method]['domain_accuracy'].get(domain, 0)
                f.write(f"{acc:.4f} | ")
            f.write("\n")
        f.write("\n")
        
        # Category analysis
        all_categories = set()
        for method_metrics in metrics.values():
            all_categories.update(method_metrics['category_accuracy'].keys())
        
        if all_categories:
            f.write("## Category Analysis\n\n")
            
            f.write("| Category | " + " | ".join(method_names) + " |\n")
            f.write("|----------|" + "|".join(["-" * len(name) for name in method_names]) + "|\n")
            
            for category in sorted(all_categories):
                f.write(f"| {category} | ")
                for method in method_names:
                    acc = metrics[method]['category_accuracy'].get(category, 0)
                    f.write(f"{acc:.4f} | ")
                f.write("\n")
            f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("The comparison demonstrates that the MultimodalMCTS approach ")
        
        if 'mcts' in metrics and best_method == 'mcts':
            f.write("outperforms all baseline methods. ")
            f.write("This confirms that the combination of Monte Carlo Tree Search with effective reward signals ")
            f.write("enables more accurate and efficient tool use strategies for multimodal question answering.\n\n")
        elif 'mcts' in metrics:
            f.write("performs competitively with baseline methods. ")
            f.write("While not achieving the highest accuracy, the approach demonstrates the potential ")
            f.write("of reinforcement learning techniques for discovering effective tool use strategies.\n\n")
        else:
            f.write("provides valuable insights into the effectiveness of different tool use strategies ")
            f.write("for multimodal question answering.\n\n")
    
    logger.info(f"Comparison summary saved to {summary_path}")
    return summary_path

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load or initialize system
    system = import_mcts_system(config)
    
    # Setup baseline methods
    baseline_methods = setup_baseline_methods(config, system)
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Use pre-computed results or evaluate
    if args.results_dir and os.path.exists(args.results_dir):
        logger.info(f"Loading pre-computed results from {args.results_dir}...")
        
        results = {}
        for method_name in baseline_methods.keys():
            result_path = os.path.join(args.results_dir, f"{method_name}_results.json")
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    results[method_name] = json.load(f)
                logger.info(f"Loaded results for {method_name} from {result_path}")
    else:
        # Evaluate all methods
        results = evaluate_methods(baseline_methods, dataset, args.num_examples, config, args.output_dir)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualizations
    visualization_paths = generate_comparison_visualizations(metrics, args.output_dir, baseline_methods)
    
    # Generate summary
    summary_path = generate_comparison_summary(metrics, visualization_paths, args.output_dir)
    
    logger.info(f"All comparison results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 