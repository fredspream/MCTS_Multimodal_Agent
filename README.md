# MultimodalMCTS: Monte Carlo Tree Search for Multimodal Question Answering

![MultimodalMCTS](https://img.shields.io/badge/MultimodalMCTS-v0.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

MultimodalMCTS is a research framework that implements Monte Carlo Tree Search (MCTS) for multimodal question answering. The system integrates various AI models and tools like image captioning, OCR, and reasoning to answer complex questions that require understanding of both text and images.

## Features

- **Monte Carlo Tree Search (MCTS)** exploration of reasoning paths
- **Multimodal understanding** with LLaVA for image understanding
- **LLaMA-based reasoning** with ReAct framework
- **OCR integration** for text extraction from images
- **ScienceQA dataset** integration for evaluation
- **Comprehensive evaluation tools** with detailed reports and visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended but not required)
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/MultimodalMCTS.git
cd MultimodalMCTS
```

### Step 2: Create a virtual environment

#### On Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the ScienceQA dataset

```bash
python scripts/download_scienceqa.py
```

## Project Structure

```
MultimodalMCTS/
├── data/
│   └── scienceqa/       # ScienceQA dataset
├── src/
│   ├── data/            # Dataset handling
│   ├── mcts/            # MCTS implementation
│   ├── models/          # LLaVA and LLaMA models
│   └── tools/           # Tools like OCR
├── evaluate_scienceqa.py  # Main evaluation script
├── evaluate_simplified.py # Simplified evaluation script
├── evaluate_robust.py     # Robust evaluation with error handling
└── requirements.txt       # Project dependencies
```

## Usage

### Running the evaluation

The system can be evaluated on the ScienceQA dataset using the following command:

```bash
python evaluate_robust.py --num_examples 20 --split val
```

For a simplified evaluation (without loading large models):

```bash
python evaluate_robust.py --num_examples 20 --split val --use_simplified
```

### Command-line arguments

- `--dataset_path`: Path to ScienceQA dataset (default: 'data/scienceqa')
- `--split`: Dataset split to evaluate on (choices: 'train', 'val', 'test', default: 'val')
- `--num_examples`: Number of examples to evaluate (default: 20)
- `--output_dir`: Directory to save evaluation results (default: 'evaluation_results')
- `--use_simplified`: Use simplified MCTS QA system (doesn't require loading large models)

### Example output

The evaluation will generate:
- Detailed JSON results with each question, answer, and correctness
- Visualization of tool usage
- Confidence distribution charts
- A comprehensive Markdown report with analysis

## System Components

### MCTS Implementation

The MCTS algorithm explores different reasoning paths by:
1. Selecting tools to gather information (image captioning, OCR)
2. Simulating forward to find answers
3. Backpropagating rewards
4. Selecting the most promising reasoning path

### Models

- **LLaVA**: Used for image understanding and caption generation
- **LLaMA with ReAct**: Used for reasoning and tool selection

### Tools

- **OCR Tool**: Extracts text from images using Tesseract
- **Caption Tool**: Generates image descriptions
- **Answer Tool**: Provides final answers based on gathered information

## Evaluation Results

The system is evaluated on the ScienceQA dataset, which contains multimodal questions from elementary to high school level.

Evaluation metrics include:
- Overall accuracy
- Tool usage distribution
- Confidence calibration
- Detailed analysis of correct and incorrect answers

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_mcts,
  author = {Your Name},
  title = {MultimodalMCTS: Monte Carlo Tree Search for Multimodal Question Answering},
  year = {2025},
  url = {https://github.com/yourusername/MultimodalMCTS}
}
```

## Acknowledgments

- ScienceQA dataset creators
- LLaVA and LLaMA model developers
- Contributors to the MCTS algorithm implementations 