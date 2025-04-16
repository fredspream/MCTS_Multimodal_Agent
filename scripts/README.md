# MultimodalMCTS Reward Signal Analysis

This directory contains scripts for analyzing the reward signal in the MultimodalMCTS system.

## Scripts

### `analyze_reward_signal.py`

This script analyzes the reward signal in MCTS trajectories to evaluate its effectiveness in guiding the search process and its correlation with successful outcomes.

#### Usage

```bash
python scripts/analyze_reward_signal.py \
    --config config/mcts_config.yaml \
    --trajectories_dir data/mcts_trajectories \
    --num_examples 50 \
    --output_dir results/reward_analysis
```

#### Parameters

- `--config`: Path to the MCTS configuration file (required)
- `--trajectories_dir`: Directory containing saved MCTS trajectories (default: "mcts_trajectories")
- `--num_examples`: Number of examples to analyze (default: 50)
- `--output_dir`: Directory to save analysis results (default: "reward_analysis_results")

#### Outputs

The script generates the following outputs in the specified output directory:

1. **Reward Distribution Visualizations**:
   - `reward_distribution.png`: Overall distribution of rewards
   - `reward_by_depth.png`: Mean reward by search depth
   - `reward_by_tool.png`: Mean reward by tool type

2. **Correlation Analysis**:
   - `reward_correlation.png`: Correlation between reward and final success

3. **Summary Reports**:
   - `reward_analysis_summary.json`: JSON file with detailed analysis results
   - `reward_analysis_report.txt`: Human-readable report with key findings and conclusions

## Running the Analysis

1. Ensure you have generated MCTS trajectories by running your experiments first
2. Run the analysis script with the appropriate parameters
3. Review the generated visualizations and reports to understand reward signal effectiveness 