# 🦀🐍 Rust vs Python ML Benchmarks

A comprehensive benchmarking system comparing Rust and Python machine learning frameworks across classical ML, deep learning, reinforcement learning, and large language model tasks.

![Benchmark Overview](docs/Figure1.pdf)

## 🎯 Overview

This project provides a scientifically rigorous comparison between Rust and Python ML ecosystems, implementing benchmarks across four major categories:

- **Classical ML**: Regression, SVM, Clustering
- **Deep Learning**: CNN (LeNet, ResNet18, Simple CNN), RNN (LSTM, GRU)  
- **Reinforcement Learning**: DQN, Policy Gradient
- **Large Language Models**: GPT-2, BERT transformers

## 🚀 Quick Start

```bash
# Setup environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run a benchmark
python src/python/classical_ml/regression_benchmark.py --dataset boston_housing --algorithm linear --mode training

# Analyze results
python scripts/analysis/analyze_results_simple.py --results-dir results
```

## 📊 What You Get

### Comprehensive Metrics
- **Performance**: Training time, inference latency, throughput
- **Resources**: Memory usage, CPU/GPU utilization  
- **Quality**: Accuracy, F1-score, loss, RMSE

### Sample Output
```
🏆 BENCHMARK RESULTS SUMMARY
Language Dataset      Architecture Device Time(s)  Memory(MB)   Accuracy(%)
Rust     MNIST        LeNet        GPU    0.463    834.9        10.50
Python   MNIST        LeNet        GPU    159.916  2611.5       99.39
Rust     CIFAR10      ResNet18     GPU    0.597    848.1        10.00
Python   CIFAR10      ResNet18     GPU    306.750  2874.7       77.91

🚀 Rust is 115.76x FASTER
💾 Rust uses 2.89x LESS memory
```

## 🏗️ Repository Structure

```
├── src/
│   ├── python/           # Python benchmarks (scikit-learn, PyTorch, transformers)
│   └── rust/             # Rust benchmarks (linfa, tch, candle)
├── scripts/
│   └── analysis/         # Analysis and visualization tools
├── results/              # All benchmark outputs and reports
├── workflows/            # Nextflow orchestration
├── config/               # Benchmark configurations
└── docs/                 # Documentation and figures
```

## 📈 Supported Benchmarks

### Classical Machine Learning
- **Datasets**: Iris, Wine, Breast Cancer, Boston Housing, California Housing
- **Algorithms**: Linear/Ridge Regression, SVM, K-Means Clustering
- **Frameworks**: scikit-learn (Python) vs linfa (Rust)

### Deep Learning
- **Datasets**: MNIST, CIFAR-10, CIFAR-100, Synthetic
- **Models**: LeNet, Simple CNN, ResNet18, VGG16, MobileNet
- **Frameworks**: PyTorch (Python) vs tch (Rust)

### Reinforcement Learning  
- **Environments**: CartPole, MountainCar, Synthetic
- **Algorithms**: DQN, DDQN, Policy Gradient
- **Frameworks**: stable-baselines3 (Python) vs custom (Rust)

### Large Language Models
- **Tasks**: Text generation, sentiment analysis, Q&A
- **Models**: GPT-2, BERT, DistilBERT
- **Frameworks**: transformers (Python) vs candle (Rust)

## 🔧 Advanced Usage

### Individual Benchmarks
```bash
# Classical ML
python src/python/classical_ml/svm_benchmark.py --dataset iris --algorithm svc --mode training

# Deep Learning  
python src/python/deep_learning/cnn_benchmark.py --model resnet18 --dataset cifar10 --device gpu

# Run Rust equivalent (after building)
cd src/rust/classical_ml/svm_benchmark && cargo run -- --dataset iris --algorithm svc --mode training
```

### Workflow Orchestration
```bash
# Complete pipeline
nextflow run main.nf

# Smoke test
nextflow run workflows/smoke.nf
```

### Results Analysis
```bash
# Simple analysis (no dependencies)
python scripts/analysis/analyze_results_simple.py --results-dir results

# Advanced analysis with pandas
python scripts/analysis/analyze_benchmark_results.py --results-dir results
```

## 📊 Results & Visualization

All outputs are saved to `results/`:
- Individual benchmark JSON files
- Aggregated CSV exports (`benchmark_results_complete.csv`)
- Summary reports (`benchmark_summary.txt`)
- Smoke test results from `smoke_results_*/`

![Performance Comparison](docs/Figure2.pdf)

## 🧪 Testing & Validation

```bash
# Run test suite
python tests/test_benchmark_system.py

# Validate specific benchmark
python tests/test_rust_fixed_concept.py
```

## 📚 Documentation

- **[USERGUIDE.md](USERGUIDE.md)** - Detailed setup, usage examples, and troubleshooting
- **[SPECS.md](SPECS.md)** - Complete technical specifications and methodology
- **[docs/](docs/)** - Additional documentation and research papers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the analysis to ensure functionality: `python scripts/analysis/analyze_results_simple.py --results-dir results`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Python ML Community** - For mature frameworks (scikit-learn, PyTorch, transformers)
- **Rust ML Community** - For performance-focused implementations (linfa, tch, candle)
- **Research Community** - For the methodological foundations

---

**Status**: ✅ Production Ready | **Last Updated**: January 2025

For detailed instructions and examples, see **[USERGUIDE.md](USERGUIDE.md)**