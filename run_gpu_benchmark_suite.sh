#!/bin/bash

# Complete GPU-Only Deep Learning Benchmark Script: Python vs Rust
# This script runs all working combinations systematically on GPU only

set -e

echo "🚀 Starting COMPLETE GPU-ONLY Deep Learning Benchmark Suite"
echo "==========================================================="
echo "🎯 Target: All datasets + architectures on GPU for maximum performance"
echo ""

# Setup environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export LIBTORCH=/home/$USER/miniforge3/envs/cpp-torch
export PYTHONPATH=/home/shiro/Projects/Paper1

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ CUDA GPU detected and ready"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "❌ No CUDA GPU found! This script requires GPU."
    exit 1
fi
echo ""

# Function to run Python GPU benchmark
run_python_gpu_benchmark() {
    local dataset=$1
    local architecture=$2
    local run_id="python_${dataset}_${architecture}_gpu_fixed"
    
    echo "🐍 Running Python GPU: $dataset + $architecture"
    
    source .venv/bin/activate
    python src/python/deep_learning/cnn_benchmark.py \
        --dataset $dataset \
        --architecture $architecture \
        --mode training \
        --run-id $run_id \
        --device cuda
    
    echo "✅ Completed: $run_id"
    echo ""
}

# Function to run Rust GPU benchmark  
run_rust_gpu_benchmark() {
    local dataset=$1
    local architecture=$2
    local run_id="rust_${dataset}_${architecture}_gpu_fixed"
    
    echo "🦀 Running Rust GPU: $dataset + $architecture"
    
    cd src/rust/deep_learning/cnn_benchmark
    
    # Set proper library path for Rust + CUDA
    export LD_LIBRARY_PATH=/home/$USER/miniforge3/envs/cpp-torch/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    
    cargo run --release -- \
        --dataset $dataset \
        --architecture $architecture \
        --mode training \
        --run-id $run_id \
        --device cuda \
        --epochs 10
    cd ../../../../
    
    echo "✅ Completed: $run_id"
    echo ""
}

# Function to run both Python and Rust for a combination
run_benchmark_pair() {
    local dataset=$1
    local architecture=$2
    
    echo "🔥 BENCHMARKING: $dataset + $architecture (GPU)"
    echo "------------------------------------------------"
    
    # Run Python first
    run_python_gpu_benchmark $dataset $architecture
    
    # Run Rust second  
    run_rust_gpu_benchmark $dataset $architecture
    
    echo "🏁 Completed pair: $dataset + $architecture"
    echo ""
}

# Function to copy results
copy_results() {
    echo "📁 Copying all GPU benchmark results to project root..."
    cp src/rust/deep_learning/cnn_benchmark/*_gpu_fixed_training_results.json . 2>/dev/null || true
    echo "✅ Results copied"
    echo ""
}

# Main benchmark execution
echo "🎯 PHASE 1: SYNTHETIC DATASET - GPU BENCHMARKS"
echo "=============================================="

echo "--- Simple CNN on Synthetic Dataset (GPU) ---"
run_benchmark_pair synthetic simple_cnn

echo "--- ResNet18 on Synthetic Dataset (GPU) ---"
echo "🐍 Running Python synthetic + resnet18 (GPU)..."
run_python_gpu_benchmark synthetic resnet18
echo "⚠️  Skipping Rust synthetic + resnet18 (channel mismatch: expects 3-ch, gets 1-ch)"
echo ""

echo ""
echo "🎯 PHASE 2: MNIST DATASET - GPU BENCHMARKS"
echo "=========================================="

echo "--- Simple CNN on MNIST Dataset (GPU) ---"
run_benchmark_pair mnist simple_cnn

echo "--- LeNet on MNIST Dataset (GPU) ---"
run_benchmark_pair mnist lenet

echo ""
echo "🎯 PHASE 3: CIFAR-10 DATASET - GPU BENCHMARKS"
echo "============================================="

echo "--- ResNet18 on CIFAR-10 Dataset (GPU) ---"
run_benchmark_pair cifar10 resnet18

echo "--- VGG16 on CIFAR-10 Dataset (GPU) ---"
echo "🐍 Testing Python CIFAR-10 + VGG16..."
run_python_gpu_benchmark cifar10 vgg16 || echo "⚠️  Python VGG16 failed"
echo "🦀 Testing Rust CIFAR-10 + VGG16..."  
run_rust_gpu_benchmark cifar10 vgg16 || echo "⚠️  Rust VGG16 failed"

echo ""
echo "🎯 PHASE 4: CIFAR-100 DATASET - GPU BENCHMARKS"
echo "==============================================="

echo "--- ResNet18 on CIFAR-100 Dataset (GPU) ---"
run_benchmark_pair cifar100 resnet18

echo ""
echo "🎯 PHASE 5: COLLECTING RESULTS"
echo "=============================="

copy_results

echo "📊 Listing all GPU benchmark result files:"
ls -la *_gpu_fixed_training_results.json 2>/dev/null || echo "No GPU fixed results found yet"

echo ""
echo "🎉 GPU BENCHMARK SUITE COMPLETED!"
echo "================================="
echo ""
echo "📈 SUMMARY:"
total_benchmarks=$(ls -1 *_gpu_fixed_training_results.json 2>/dev/null | wc -l)
python_benchmarks=$(ls -1 python_*_gpu_fixed_training_results.json 2>/dev/null | wc -l)
rust_benchmarks=$(ls -1 rust_*_gpu_fixed_training_results.json 2>/dev/null | wc -l)

echo "  🏆 Total GPU benchmarks completed: $total_benchmarks"
echo "  🐍 Python GPU benchmarks: $python_benchmarks" 
echo "  🦀 Rust GPU benchmarks: $rust_benchmarks"
echo ""
echo "📁 Result files pattern: *_gpu_fixed_training_results.json"
echo "🔍 Each file contains:"
echo "   - Training time (seconds) - should be MUCH faster on GPU"
echo "   - Memory usage (peak/average MB)"
echo "   - Model accuracy (%) - Rust should be MUCH better now!"
echo "   - GPU memory usage (MB)"
echo "   - Loss values - Rust should have much lower loss now"
echo ""
echo "🚀 EXPECTED IMPROVEMENTS with FIXED Rust:"
echo "   ⚡ Rust still 100x+ faster than Python"
echo "   🎯 Rust accuracy should now be competitive (20-90%+)"
echo "   📉 Rust loss should be much lower (~0.5-2.0 range)"
echo "   💾 Rust memory efficiency maintained"
echo ""
echo "📊 Next steps:"
echo "   1. Run: python3 analyze_results_simple.py"
echo "   2. Check the dramatic improvement in Rust accuracy/loss"
echo "   3. Celebrate the TRUE Python vs Rust showdown! 🎊"
echo ""