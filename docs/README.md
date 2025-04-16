# Multimodal MCTS for Question Answering

This project implements a Monte Carlo Tree Search (MCTS) approach for multimodal question answering tasks. The system uses LLaVA for image captioning, Tesseract for OCR, and a ReAct-style LLaMA model for reasoning.

## Project Structure

```
MultimodalMCTS/
├── configs/            # Configuration files
├── data/               # Dataset directory (will be populated by download script)
├── docs/               # Documentation
├── examples/           # Example usage scripts
├── models/             # Model files (will be populated by download script)
├── scripts/            # Utility scripts
├── src/                # Source code
│   ├── data/           # Dataset utilities
│   ├── mcts/           # MCTS implementation
│   ├── models/         # Model integrations
│   ├── tools/          # Tool implementations
│   └── multimodal_mcts_qa.py  # Main MCTS QA system
├── main.py             # Main entry point for evaluation
├── requirements.txt    # Dependencies
├── README.md           # Project overview
└── run.py              # End-to-end pipeline script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/MultimodalMCTS.git
cd MultimodalMCTS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download models and data:
```bash
python scripts/download_models.py --download_scienceqa
```

## Usage

### Quick Start

For a quick end-to-end run of the pipeline:

```bash
python run.py --num_examples 5 --output_dir results
```

### Running Individual Steps

1. Download datasets and models:
```bash
python scripts/download_models.py --download_scienceqa --download_llava --download_llama
```

2. Run evaluation on ScienceQA:
```bash
python main.py --config configs/default_config.yaml --num_examples 10
```

3. Run with a single example:
```bash
python examples/single_example.py
```

## Configuration

The system can be configured through YAML configuration files in the `configs` directory. Key configurations include:

- Model settings (LLaVA and LLaMA parameters)
- OCR settings (Tesseract configuration)
- MCTS parameters (time limit, iterations, exploration weight)
- Dataset settings
- Evaluation parameters

See `configs/default_config.yaml` for details.

## Components

### MCTS Implementation

The MCTS algorithm is implemented in `src/mcts/` with these key components:

- `MCTSNode`: Base node class for the search tree
- `MultimodalQANode`: Specialized node for multimodal QA
- `MCTS`: Main implementation of the search algorithm

### Models

- `LLaVAModel`: Integration with LLaVA for image captioning and visual QA
- `LLaMAReAct`: ReAct-style integration with LLaMA for tool-aware reasoning

### Tools

- `OCRTool`: Wrapper for Tesseract OCR to extract text from images

### Dataset

- `ScienceQADataset`: Utility for loading and processing the ScienceQA dataset

## How It Works

The system follows these steps to answer questions:

1. Given a question and image, the MCTS algorithm explores different reasoning paths.
2. At each step, it can choose to:
   - Generate a caption for the image (using LLaVA)
   - Extract text from the image (using Tesseract OCR)
   - Generate a final answer (using LLaMA)
3. The search uses the outcomes of previous steps to guide future exploration.
4. Finally, it selects the most promising path and generates an answer.

For more details on the MCTS approach, see `docs/mcts_approach.md`.

## Customization

To adapt this system to other multimodal tasks:

1. Create a new node class extending `MCTSNode` for your task
2. Customize the available actions and state representation
3. Implement task-specific tools
4. Modify the reward function to align with your task objectives

## License

MIT License 