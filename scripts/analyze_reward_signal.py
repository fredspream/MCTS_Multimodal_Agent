#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyzes the reward signal in MCTS trajectories to evaluate effectiveness.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml

# Add the project root to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze MCTS reward signal effectiveness"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to MCTS configuration file",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="mcts_trajectories",
        help="Directory containing MCTS trajectories",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="Number of examples to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reward_analysis_results",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def import_mcts_system(config: Dict[str, Any]) -> None:
    """Import the MCTS system based on configuration."""
    # This function is a placeholder for importing the actual MCTS system
    # In a real implementation, you might import specific modules needed for analysis
    logger.info("Importing MCTS system components")
    pass


def load_trajectories(
    trajectories_dir: str, num_examples: int
) -> List[Dict[str, Any]]:
    """Load MCTS trajectories from JSON files in the specified directory."""
    logger.info(f"Loading up to {num_examples} trajectories from {trajectories_dir}")
    
    trajectory_files = list(Path(trajectories_dir).glob("*.json"))
    if not trajectory_files:
        raise FileNotFoundError(f"No JSON files found in {trajectories_dir}")
    
    # Load up to num_examples trajectory files
    trajectories = []
    for file_path in trajectory_files[:num_examples]:
        try:
            with open(file_path, "r") as f:
                trajectory = json.load(f)
                trajectories.append(trajectory)
        except Exception as e:
            logger.error(f"Error loading trajectory file {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def analyze_reward_distribution(
    trajectories: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze the distribution of rewards across trajectories."""
    logger.info("Analyzing reward distribution")
    
    # Collect all rewards
    all_rewards = []
    rewards_by_depth = defaultdict(list)
    rewards_by_tool = defaultdict(list)
    
    for trajectory in trajectories:
        # Extract nodes and their rewards
        for node in trajectory.get("nodes", []):
            reward = node.get("reward", 0)
            depth = node.get("depth", 0)
            tool_type = node.get("action", {}).get("tool_type", "unknown")
            
            all_rewards.append(reward)
            rewards_by_depth[depth].append(reward)
            rewards_by_tool[tool_type].append(reward)
    
    # Compute statistics
    reward_stats = {
        "mean": np.mean(all_rewards) if all_rewards else 0,
        "median": np.median(all_rewards) if all_rewards else 0,
        "std": np.std(all_rewards) if all_rewards else 0,
        "min": np.min(all_rewards) if all_rewards else 0,
        "max": np.max(all_rewards) if all_rewards else 0,
        "count": len(all_rewards),
        "by_depth": {
            depth: {
                "mean": np.mean(rewards),
                "count": len(rewards),
            }
            for depth, rewards in rewards_by_depth.items()
        },
        "by_tool": {
            tool: {
                "mean": np.mean(rewards),
                "count": len(rewards),
            }
            for tool, rewards in rewards_by_tool.items()
        },
    }
    
    return reward_stats


def analyze_reward_correlation(
    trajectories: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze the correlation between rewards and success."""
    logger.info("Analyzing correlation between rewards and success")
    
    # Prepare data for correlation analysis
    data = []
    for trajectory in trajectories:
        # Extract the final outcome of the trajectory
        success = trajectory.get("success", False)
        
        # Get max reward and average reward
        rewards = [node.get("reward", 0) for node in trajectory.get("nodes", [])]
        if not rewards:
            continue
            
        max_reward = max(rewards)
        avg_reward = np.mean(rewards)
        final_reward = rewards[-1] if rewards else 0
        
        data.append({
            "success": 1 if success else 0,
            "max_reward": max_reward,
            "avg_reward": avg_reward,
            "final_reward": final_reward,
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Calculate correlations if we have data
    correlations = {}
    if not df.empty:
        correlations = {
            "max_reward_success": df["max_reward"].corr(df["success"]),
            "avg_reward_success": df["avg_reward"].corr(df["success"]),
            "final_reward_success": df["final_reward"].corr(df["success"]),
        }
    
    # Calculate success rates by reward quantiles
    correlation_analysis = {
        "correlations": correlations,
        "data": df.to_dict("records"),
    }
    
    return correlation_analysis


def analyze_reward_effectiveness(
    trajectories: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze how effectively the reward signal guides the MCTS search."""
    logger.info("Analyzing reward signal effectiveness in guiding search")
    
    # Track how often higher reward nodes are selected
    reward_guided_selections = 0
    total_selections = 0
    
    # Track reward gradient during search
    reward_gradients = []
    
    for trajectory in trajectories:
        nodes = trajectory.get("nodes", [])
        
        # Analyze selection patterns
        for i, node in enumerate(nodes):
            if i == 0:
                continue
                
            # Check if this node's parent had multiple children
            parent_id = node.get("parent_id")
            siblings = [n for n in nodes if n.get("parent_id") == parent_id]
            
            if len(siblings) <= 1:
                continue
                
            # Check if the selected node had higher reward than its siblings
            node_reward = node.get("reward", 0)
            sibling_rewards = [n.get("reward", 0) for n in siblings if n != node]
            
            if sibling_rewards and node_reward > max(sibling_rewards):
                reward_guided_selections += 1
            
            total_selections += 1
        
        # Calculate reward gradient (how rewards improve over depth)
        rewards_by_depth = defaultdict(list)
        for node in nodes:
            depth = node.get("depth", 0)
            reward = node.get("reward", 0)
            rewards_by_depth[depth].append(reward)
        
        # Calculate average reward at each depth
        avg_rewards = {
            depth: np.mean(rewards) 
            for depth, rewards in rewards_by_depth.items()
        }
        
        # Calculate gradient (improvement between consecutive depths)
        depths = sorted(avg_rewards.keys())
        for i in range(1, len(depths)):
            prev_depth = depths[i-1]
            curr_depth = depths[i]
            gradient = avg_rewards[curr_depth] - avg_rewards[prev_depth]
            reward_gradients.append(gradient)
    
    # Compute effectiveness metrics
    effectiveness = {
        "guided_selection_rate": reward_guided_selections / total_selections if total_selections > 0 else 0,
        "avg_reward_gradient": np.mean(reward_gradients) if reward_gradients else 0,
        "reward_gradient_std": np.std(reward_gradients) if reward_gradients else 0,
    }
    
    return effectiveness


def plot_reward_distributions(
    reward_stats: Dict[str, Any], output_dir: str
) -> None:
    """Generate visualizations for reward distributions."""
    logger.info("Generating reward distribution visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot overall reward distribution
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract all rewards from nested structure for plotting
    # This is just a placeholder - you'll need to adapt this based on your actual data structure
    all_rewards = []
    for trajectory in reward_stats.get("data", []):
        all_rewards.append(trajectory.get("max_reward", 0))
        all_rewards.append(trajectory.get("avg_reward", 0))
        all_rewards.append(trajectory.get("final_reward", 0))
    
    if all_rewards:
        sns.histplot(all_rewards, kde=True)
        plt.title("Distribution of Rewards")
        plt.xlabel("Reward Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, "reward_distribution.png"))
        plt.close()
    
    # Plot mean reward by depth
    depths = sorted(reward_stats.get("by_depth", {}).keys())
    mean_rewards = [reward_stats["by_depth"][depth]["mean"] for depth in depths]
    
    if depths and mean_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(depths, mean_rewards, marker='o')
        plt.title("Mean Reward by Search Depth")
        plt.xlabel("Depth")
        plt.ylabel("Mean Reward")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "reward_by_depth.png"))
        plt.close()
    
    # Plot mean reward by tool type
    tools = list(reward_stats.get("by_tool", {}).keys())
    tool_rewards = [reward_stats["by_tool"][tool]["mean"] for tool in tools]
    
    if tools and tool_rewards:
        plt.figure(figsize=(12, 6))
        plt.bar(tools, tool_rewards)
        plt.title("Mean Reward by Tool Type")
        plt.xlabel("Tool Type")
        plt.ylabel("Mean Reward")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_by_tool.png"))
        plt.close()


def plot_reward_correlation(
    correlation_analysis: Dict[str, Any], output_dir: str
) -> None:
    """Generate visualizations for reward-success correlations."""
    logger.info("Generating reward-success correlation visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from correlation data
    df = pd.DataFrame(correlation_analysis.get("data", []))
    
    if df.empty:
        logger.warning("No data available for correlation plots")
        return
    
    # Plot success rate by reward quantile
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different reward metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot for max reward
    if "max_reward" in df.columns and "success" in df.columns:
        df["max_reward_quantile"] = pd.qcut(df["max_reward"], 5, labels=False)
        success_by_max = df.groupby("max_reward_quantile")["success"].mean()
        
        axes[0].bar(success_by_max.index, success_by_max.values)
        axes[0].set_title(f"Success Rate by Max Reward Quantile\nCorr: {correlation_analysis['correlations'].get('max_reward_success', 'N/A'):.2f}")
        axes[0].set_xlabel("Max Reward Quantile")
        axes[0].set_ylabel("Success Rate")
    
    # Plot for avg reward
    if "avg_reward" in df.columns and "success" in df.columns:
        df["avg_reward_quantile"] = pd.qcut(df["avg_reward"], 5, labels=False)
        success_by_avg = df.groupby("avg_reward_quantile")["success"].mean()
        
        axes[1].bar(success_by_avg.index, success_by_avg.values)
        axes[1].set_title(f"Success Rate by Avg Reward Quantile\nCorr: {correlation_analysis['correlations'].get('avg_reward_success', 'N/A'):.2f}")
        axes[1].set_xlabel("Avg Reward Quantile")
        axes[1].set_ylabel("Success Rate")
    
    # Plot for final reward
    if "final_reward" in df.columns and "success" in df.columns:
        df["final_reward_quantile"] = pd.qcut(df["final_reward"], 5, labels=False)
        success_by_final = df.groupby("final_reward_quantile")["success"].mean()
        
        axes[2].bar(success_by_final.index, success_by_final.values)
        axes[2].set_title(f"Success Rate by Final Reward Quantile\nCorr: {correlation_analysis['correlations'].get('final_reward_success', 'N/A'):.2f}")
        axes[2].set_xlabel("Final Reward Quantile")
        axes[2].set_ylabel("Success Rate")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_correlation.png"))
    plt.close()


def generate_report(
    reward_stats: Dict[str, Any],
    correlation_analysis: Dict[str, Any],
    effectiveness: Dict[str, Any],
    output_dir: str,
) -> None:
    """Generate a comprehensive report of the reward signal analysis."""
    logger.info("Generating analysis report")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all analysis results
    report_data = {
        "reward_statistics": reward_stats,
        "correlation_analysis": correlation_analysis,
        "effectiveness_metrics": effectiveness,
    }
    
    # Save detailed JSON report
    with open(os.path.join(output_dir, "reward_analysis_summary.json"), "w") as f:
        json.dump(report_data, f, indent=2)
    
    # Generate human-readable text report
    report_text = [
        "# MCTS Reward Signal Analysis Report",
        "",
        "## 1. Reward Distribution Statistics",
        f"Mean Reward: {reward_stats.get('mean', 'N/A'):.4f}",
        f"Median Reward: {reward_stats.get('median', 'N/A'):.4f}",
        f"Standard Deviation: {reward_stats.get('std', 'N/A'):.4f}",
        f"Min Reward: {reward_stats.get('min', 'N/A'):.4f}",
        f"Max Reward: {reward_stats.get('max', 'N/A'):.4f}",
        f"Total Samples: {reward_stats.get('count', 'N/A')}",
        "",
        "## 2. Reward-Success Correlation",
    ]
    
    # Add correlation information
    correlations = correlation_analysis.get("correlations", {})
    report_text.extend([
        f"Max Reward to Success Correlation: {correlations.get('max_reward_success', 'N/A'):.4f}",
        f"Avg Reward to Success Correlation: {correlations.get('avg_reward_success', 'N/A'):.4f}",
        f"Final Reward to Success Correlation: {correlations.get('final_reward_success', 'N/A'):.4f}",
        "",
        "## 3. Reward Effectiveness Metrics",
        f"Guided Selection Rate: {effectiveness.get('guided_selection_rate', 'N/A'):.4f}",
        f"Average Reward Gradient: {effectiveness.get('avg_reward_gradient', 'N/A'):.4f}",
        f"Reward Gradient Std Dev: {effectiveness.get('reward_gradient_std', 'N/A'):.4f}",
        "",
        "## 4. Conclusions",
    ])
    
    # Add conclusions based on analysis
    # Correlation interpretation
    max_corr = correlations.get('max_reward_success', 0)
    if abs(max_corr) < 0.3:
        corr_conclusion = "Weak correlation between rewards and success"
    elif abs(max_corr) < 0.7:
        corr_conclusion = "Moderate correlation between rewards and success"
    else:
        corr_conclusion = "Strong correlation between rewards and success"
    
    # Effectiveness interpretation
    guided_rate = effectiveness.get('guided_selection_rate', 0)
    if guided_rate < 0.5:
        effect_conclusion = "Reward signal has limited influence on search direction"
    elif guided_rate < 0.8:
        effect_conclusion = "Reward signal moderately guides search direction"
    else:
        effect_conclusion = "Reward signal strongly guides search direction"
    
    report_text.extend([
        corr_conclusion,
        effect_conclusion,
        f"Overall, the reward signal {'appears effective' if max_corr > 0.5 and guided_rate > 0.6 else 'may need improvement'} in guiding the MCTS search process.",
    ])
    
    # Save text report
    with open(os.path.join(output_dir, "reward_analysis_report.txt"), "w") as f:
        f.write("\n".join(report_text))


def main():
    """Main function to analyze MCTS reward signal."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Import MCTS system
    import_mcts_system(config)
    
    # Load trajectories
    trajectories = load_trajectories(args.trajectories_dir, args.num_examples)
    
    # Analyze reward distribution
    reward_stats = analyze_reward_distribution(trajectories)
    
    # Analyze correlation with success
    correlation_analysis = analyze_reward_correlation(trajectories)
    
    # Analyze reward effectiveness
    effectiveness = analyze_reward_effectiveness(trajectories)
    
    # Generate visualizations
    plot_reward_distributions(reward_stats, args.output_dir)
    plot_reward_correlation(correlation_analysis, args.output_dir)
    
    # Generate report
    generate_report(
        reward_stats,
        correlation_analysis,
        effectiveness,
        args.output_dir,
    )
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 