#!/bin/bash

# Complete Deep Learning Benchmark Script: Python vs Rust
# This script runs all working combinations systematically

set -e

echo "🚀 Starting Complete Deep Learning Benchmark Suite"
echo "=============================================="

# Setup environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBTORCH=/home/$USER/miniforge3/envs/cpp-torch
export PYTHONPATH=/home/shiro/Projects/Paper1

# Function to run Python benchmark
run_python_benchmark() {
    local dataset=$1
    local architecture=$2
    local device=$3
    local run_id="python_${dataset}_${architecture}_${device}_complete"
    
    echo "📊 Running Python: $dataset + $architecture + $device"
    
    # Convert device names for Python (gpu->cuda, cpu->cpu)
    local py_device=$device
    if [ "$device" = "gpu" ]; then
        py_device="cuda"
    fi
    
    source .venv/bin/activate
    python src/python/deep_learning/cnn_benchmark.py \
        --dataset $dataset \
        --architecture $architecture \
        --mode training \
        --run-id $run_id \
        --device $py_device
    
    echo "✅ Completed: $run_id"
    echo ""
}

# Function to run Rust benchmark
run_rust_benchmark() {
    local dataset=$1
    local architecture=$2
    local device=$3
    local run_id="rust_${dataset}_${architecture}_${device}_complete"
    
    echo "🦀 Running Rust: $dataset + $architecture + $device"
    
    # Convert device names for Rust (gpu->cuda, cpu->cpu)
    local rust_device=$device
    if [ "$device" = "gpu" ]; then
        rust_device="cuda"
    fi
    
    cd src/rust/deep_learning/cnn_benchmark
    
    # Set proper library path for Rust + CUDA
    export LD_LIBRARY_PATH=/home/$USER/miniforge3/envs/cpp-torch/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    
    cargo run --release -- \
        --dataset $dataset \
        --architecture $architecture \
        --mode training \
        --run-id $run_id \
        --device $rust_device \
        --epochs 10
    cd ../../../../
    
    echo "✅ Completed: $run_id"
    echo ""
}

# Function to copy results
copy_results() {
    echo "📁 Copying all results to project root..."
    cp src/rust/deep_learning/cnn_benchmark/*_complete_training_results.json . 2>/dev/null || true
    echo "✅ Results copied"
    echo ""
}

# Main benchmark execution
echo "🎯 Phase 1: SYNTHETIC DATASET BENCHMARKS"
echo "========================================"

echo "--- Simple CNN on Synthetic Dataset ---"
run_python_benchmark synthetic simple_cnn gpu
run_python_benchmark synthetic simple_cnn cpu
run_rust_benchmark synthetic simple_cnn gpu
run_rust_benchmark synthetic simple_cnn cpu

echo "--- ResNet18 on Synthetic Dataset ---"
run_python_benchmark synthetic resnet18 gpu
# Note: Rust ResNet18 with synthetic fails due to channel mismatch (expects 3-ch, gets 1-ch)

echo ""
echo "🎯 Phase 2: CIFAR-10 DATASET BENCHMARKS"
echo "======================================="

echo "--- ResNet18 on CIFAR-10 Dataset ---"
run_python_benchmark cifar10 resnet18 gpu
run_python_benchmark cifar10 resnet18 cpu
run_rust_benchmark cifar10 resnet18 gpu
run_rust_benchmark cifar10 resnet18 cpu

echo "--- VGG16 on CIFAR-10 Dataset ---"
run_python_benchmark cifar10 vgg16 gpu || echo "⚠️  Python VGG16 failed or not implemented"
run_rust_benchmark cifar10 vgg16 gpu || echo "⚠️  Rust VGG16 failed or not implemented"

echo ""
echo "🎯 Phase 3: CIFAR-100 DATASET BENCHMARKS" 
echo "========================================="

echo "--- ResNet18 on CIFAR-100 Dataset ---"
run_python_benchmark cifar100 resnet18 gpu || echo "⚠️  Python CIFAR-100 failed"
run_rust_benchmark cifar100 resnet18 gpu || echo "⚠️  Rust CIFAR-100 failed"

echo ""
echo "🎯 Phase 4: MNIST DATASET BENCHMARKS"
echo "===================================="

echo "--- Simple CNN on MNIST Dataset ---"
run_python_benchmark mnist simple_cnn gpu || echo "⚠️  Python MNIST Simple CNN failed"
run_rust_benchmark mnist simple_cnn gpu || echo "⚠️  Rust MNIST Simple CNN failed"

echo "--- LeNet on MNIST Dataset ---"
run_python_benchmark mnist lenet gpu || echo "⚠️  Python LeNet failed"
run_rust_benchmark mnist lenet gpu || echo "⚠️  Rust LeNet failed"

echo ""
echo "🎯 Phase 5: COLLECTING RESULTS"
echo "=============================="

copy_results

echo "📊 Listing all benchmark result files:"
ls -la *_complete_training_results.json 2>/dev/null || echo "No complete results found yet"

echo ""
echo "🎉 BENCHMARK SUITE COMPLETED!"
echo "============================="
echo ""
echo "📁 Result files are saved with pattern: *_complete_training_results.json"
echo "🔍 Each file contains detailed metrics for:"
echo "   - Training time (seconds)"
echo "   - Memory usage (peak/average MB)" 
echo "   - Model accuracy (%)"
echo "   - GPU memory usage (MB)"
echo "   - Resource utilization"
echo ""
echo "📈 Next steps:"
echo "   1. Check all *_complete_training_results.json files"
echo "   2. Provide results to Claude for comprehensive analysis"
echo "   3. Generate final comparison tables and recommendations"
echo ""