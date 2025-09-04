#!/bin/bash
# Quick Benchmark Runner Script for Rust vs Python Deep Learning

set -e  # Exit on any error

PROJECT_ROOT="$(pwd)"
echo "🚀 Starting Rust vs Python Deep Learning Benchmarks"
echo "📍 Project root: $PROJECT_ROOT"

# Verify CUDA availability
echo "🔍 Checking CUDA availability..."
nvidia-smi > /dev/null || { echo "❌ NVIDIA drivers not available"; exit 1; }
echo "✅ CUDA available"

# Set common environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export LIBTORCH="/home/$USER/miniforge3/envs/cpp-torch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"

# Function to run Python benchmark
run_python_benchmark() {
    local device=$1
    local run_id="python_cnn_${device}_quick"
    
    echo "🐍 Running Python $device benchmark..."
    source .venv/bin/activate
    
    PYTHONPATH="$PROJECT_ROOT" python src/python/deep_learning/cnn_benchmark.py \
        --dataset synthetic \
        --architecture simple_cnn \
        --mode training \
        --run-id "$run_id" \
        --device "$device"
    
    echo "✅ Python $device benchmark completed"
}

# Function to run Rust benchmark  
run_rust_benchmark() {
    local device=$1
    local run_id="rust_cnn_${device}_quick"
    
    echo "🦀 Running Rust $device benchmark..."
    cd "$PROJECT_ROOT/src/rust/deep_learning/cnn_benchmark"
    
    cargo run --release -- \
        --dataset synthetic \
        --architecture simple_cnn \
        --mode training \
        --run-id "$run_id" \
        --device "$device"
    
    cd "$PROJECT_ROOT"
    echo "✅ Rust $device benchmark completed"
}

# Function to show results summary
show_results() {
    echo ""
    echo "📊 BENCHMARK RESULTS SUMMARY"
    echo "=================================================="
    
    for result_file in synthetic_simple_cnn_*_quick_training_results.json; do
        if [[ -f "$result_file" ]]; then
            echo "📄 $result_file:"
            
            # Extract key metrics using jq or fallback to grep
            if command -v jq >/dev/null 2>&1; then
                training_time=$(jq -r '.performance_metrics.training_time_seconds' "$result_file")
                accuracy=$(jq -r '.quality_metrics.accuracy' "$result_file")
                memory=$(jq -r '.resource_metrics.peak_memory_mb' "$result_file")
                device=$(jq -r '.metadata.device' "$result_file")
                
                printf "  ⏱️  Training Time: %.2fs\n" "$training_time"
                printf "  🎯 Accuracy: %.1f%%\n" "$(echo "$accuracy * 100" | bc -l)"
                printf "  💾 Peak Memory: %.0fMB\n" "$memory"
                printf "  ⚙️  Device: %s\n" "$device"
            else
                echo "  📋 Raw results (install jq for formatted display):"
                grep -E "(training_time_seconds|accuracy|peak_memory_mb|device)" "$result_file" | head -4
            fi
            echo ""
        fi
    done
    
    # Find result files
    python_gpu=$(find . -name "*python*gpu*quick*.json" | head -1)
    python_cpu=$(find . -name "*python*cpu*quick*.json" | head -1)
    rust_gpu=$(find . -name "*rust*gpu*quick*.json" | head -1)
    rust_cpu=$(find . -name "*rust*cpu*quick*.json" | head -1)
    
    if [[ -f "$python_gpu" && -f "$rust_gpu" && command -v jq >/dev/null 2>&1 ]]; then
        py_gpu_time=$(jq -r '.performance_metrics.training_time_seconds' "$python_gpu")
        rust_gpu_time=$(jq -r '.performance_metrics.training_time_seconds' "$rust_gpu")
        
        speedup=$(echo "scale=1; $py_gpu_time / $rust_gpu_time" | bc -l)
        echo "🏆 WINNER: Rust GPU is ${speedup}x faster than Python GPU!"
    fi
}

# Main execution
echo "🏁 Starting benchmark suite..."

# Run all benchmarks
run_python_benchmark "gpu"
run_python_benchmark "cpu"
run_rust_benchmark "gpu"
run_rust_benchmark "cpu"

# Show results
show_results

echo ""
echo "🎉 All benchmarks completed successfully!"
echo "📁 Result files are in the current directory with suffix '_quick_training_results.json'"
echo ""
echo "🔗 For detailed setup instructions, see: DEEP_LEARNING_BENCHMARK_GUIDE.md"