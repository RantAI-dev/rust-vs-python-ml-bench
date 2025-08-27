# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive machine learning benchmark system comparing Rust and Python frameworks across Classical ML, Deep Learning, Large Language Models, and Reinforcement Learning. The project uses Nextflow for workflow orchestration and includes both Python and Rust implementations.

## Common Commands

### Environment Setup
```bash
# Create and activate Python virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Build all Rust benchmarks
find src/rust -name "Cargo.toml" -execdir cargo build --release \;

# Setup environment (creates directories and default configs)
./scripts/setup_environment.sh
```

### Running Benchmarks
```bash
# Run complete benchmark pipeline
nextflow run main.nf

# Run specific workflow phases
nextflow run workflows/phase1_selection.nf
nextflow run workflows/phase4_benchmark.nf

# Run smoke tests (quick validation)
nextflow run workflows/smoke.nf

# Resume failed runs
nextflow run workflows/smoke.nf -resume
```

### Testing
```bash
# Run Python tests
python -m pytest tests/ -v

# Run Rust tests for all benchmarks
cargo test --all

# Run comprehensive test suite
python tests/test_benchmark_system.py
```

### Individual Benchmark Execution
```bash
# Python benchmarks
python src/python/classical_ml/regression_benchmark.py --dataset boston_housing --algorithm linear --mode training
python src/python/deep_learning/cnn_benchmark.py --dataset mnist --architecture resnet18 --mode training

# Rust benchmarks (after building)
cd src/rust/classical_ml/regression_benchmark && cargo run -- --dataset boston_housing --algorithm linear --mode training
```

## High-Level Architecture

### Workflow Orchestration
- **Nextflow**: Main workflow orchestration using DSL2
- **6-Phase Methodology**: Framework selection → Implementation → Experimentation → Benchmarking → Analysis → Assessment
- **Profiles**: Local, SLURM, Docker, Singularity execution profiles
- **Smoke Testing**: Quick validation workflow for development

### Directory Structure
- `src/python/`: Python benchmark implementations (classical_ml, deep_learning, llm, reinforcement_learning)
- `src/rust/`: Rust benchmark implementations (mirroring Python structure)
- `src/shared/`: Common schemas and utilities
- `workflows/`: Nextflow workflow definitions for each phase
- `config/`: YAML configuration files (benchmarks, frameworks, hardware)
- `scripts/`: Utility scripts for setup, validation, and analysis

### Language Implementations

#### Python Stack
- **Classical ML**: scikit-learn (1.3.2)
- **Deep Learning**: PyTorch (2.0.1) 
- **LLM**: transformers (4.30.2), Hugging Face ecosystem
- **RL**: stable-baselines3 (2.1.0), gymnasium

#### Rust Stack
- **Classical ML**: linfa (0.7.0) ecosystem
- **Deep Learning**: tch (0.13.0) - PyTorch bindings
- **LLM**: candle-transformers (0.3.3)
- **RL**: Custom implementations using tch

### Benchmark Categories
1. **Classical ML**: Regression, SVM, Clustering
2. **Deep Learning**: CNN (ResNet18, VGG16, etc.), RNN (LSTM, GRU)
3. **LLM**: GPT-2, BERT variants for text generation/classification
4. **RL**: DQN variants, Policy Gradient methods

### Configuration Management
- `config/benchmarks.yaml`: Defines datasets, algorithms, hyperparameters, metrics
- `config/frameworks.yaml`: Framework versions and dependencies
- `config/hardware.yaml`: System specifications and resource limits
- `nextflow.config`: Execution profiles and process resources

### Monitoring and Metrics
- Comprehensive resource monitoring (CPU, memory, GPU)
- Performance metrics (training time, inference latency, throughput)
- Quality metrics (accuracy, F1-score, RMSE, perplexity)
- Statistical analysis with effect sizes and significance testing

## Development Notes

### Smoke Testing
- **Status**: CNN, LLM, RL, RNN workflows are green
- **Python Classical ML**: Requires local venv setup, then use `-resume`
- Run smoke tests first to validate setup before full benchmarks

### Virtual Environment Usage
Always use a Python virtual environment when working with Python components:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Rust Development
- Each benchmark is a separate Cargo project under `src/rust/`
- Use `cargo build --release` for performance benchmarks
- All projects are part of a workspace defined in root `Cargo.toml`

### Nextflow Execution
- Default profile: `local_smoke` for development
- Use `-resume` to continue from last successful checkpoint
- Check `work/` directory for detailed process outputs
- Logs available in `.nextflow.log`

### Resource Configuration
- Process labels define resource requirements: `cpu_intensive`, `gpu_training`, `memory_intensive`, `smoke_light`
- Hardware configurations in `config/hardware.yaml`
- Execution profiles in `nextflow.config` for different environments