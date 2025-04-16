# Backup Information

## Files Moved to Backup (Date: 2025-04-15)

The following files were moved from the main directory to the `backup` directory to clean up the project structure:

### Evaluation Scripts
- `evaluate_simplified.py` - Simplified version replaced by more robust `evaluate_robust.py`
- `evaluate_scienceqa.py` - Base implementation superseded by `evaluate_robust.py`

### Demo Scripts
- `demo.py` - Basic demo functionality
- `simple_demo.py` - Simple demo implementation
- `interactive_demo.py` - Interactive demo functionality

### Setup and Run Scripts
- `setup.py` - Basic setup functionality
- `setup_and_run.py` - Setup and run combined script
- `run.py` - Basic run script
- `main.py` - Main entry point, replaced by specific scripts

### Test Scripts
- `test_python.py` - Python functionality tests
- `test_system.py` - System-level tests
- `test_all.py` - Combined test runner

## Remaining Core Files

The following essential files were kept in the main directory:

- `benchmark_scienceqa.py` - Comprehensive benchmark script for ScienceQA dataset
- `cleanup.py` - Utility for cleaning up the repository
- `evaluate_robust.py` - Robust evaluation script with error handling

## Why These Changes Were Made

Removing redundant files makes the project structure cleaner and more maintainable. By focusing on the essential files:

1. New users can more easily understand the project structure
2. Less confusion about which scripts to use for specific tasks
3. Reduced risk of using outdated or incomplete implementations

If any of the backed-up files are needed, they can be restored from the `backup` directory. 