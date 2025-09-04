#!/usr/bin/env python3
"""
Simple Benchmark Results Analysis Script (No dependencies)
Analyzes all *_complete_training_results.json files without pandas
"""

import json
import glob
import os

def load_benchmark_results():
    """Load all benchmark result files"""
    results = []
    
    pattern = "*_complete_training_results.json"
    files = glob.glob(pattern)
    
    print(f"📁 Found {len(files)} result files:")
    for file in files:
        print(f"   - {file}")
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    return results

def extract_key_metrics(result):
    """Extract key metrics from a result"""
    try:
        return {
            'language': result['language'].title(),
            'dataset': result['dataset'].title(),
            'architecture': result['metadata']['architecture'],
            'device': result['metadata'].get('device', 'Unknown').replace('Cuda(0)', 'GPU').replace('Cpu', 'CPU'),
            'training_time': result['performance_metrics']['training_time_seconds'],
            'peak_memory': result['resource_metrics']['peak_memory_mb'],
            'accuracy': result['quality_metrics']['accuracy'] * 100 if result['quality_metrics']['accuracy'] else 0,
            'gpu_memory': result['resource_metrics'].get('peak_gpu_memory_mb', 0) or 0,
            'cpu_usage': result['resource_metrics']['cpu_utilization_percent'],
            'final_loss': result['quality_metrics']['loss'],
            'run_id': result['run_id']
        }
    except KeyError as e:
        print(f"❌ Missing key {e} in result")
        return None

def print_table_header():
    """Print formatted table header"""
    print(f"{'Language':<8} {'Dataset':<12} {'Architecture':<12} {'Device':<6} {'Time(s)':<8} {'Memory(MB)':<12} {'Accuracy(%)':<12} {'GPU(MB)':<8}")
    print("-" * 90)

def print_result_row(metrics):
    """Print a formatted result row"""
    print(f"{metrics['language']:<8} {metrics['dataset']:<12} {metrics['architecture']:<12} "
          f"{metrics['device']:<6} {metrics['training_time']:<8.3f} {metrics['peak_memory']:<12.1f} "
          f"{metrics['accuracy']:<12.2f} {metrics['gpu_memory']:<8.1f}")

def analyze_results(results):
    """Perform comprehensive analysis"""
    
    print("\n🏆 COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 90)
    
    # Extract all metrics
    all_metrics = []
    for result in results:
        metrics = extract_key_metrics(result)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("❌ No valid results to analyze!")
        return
    
    print(f"\n📊 COMPLETE RESULTS TABLE ({len(all_metrics)} benchmarks)")
    print_table_header()
    
    # Sort by training time for performance ranking
    sorted_metrics = sorted(all_metrics, key=lambda x: x['training_time'])
    
    for metrics in sorted_metrics:
        print_result_row(metrics)
    
    # Performance rankings
    print(f"\n🚀 TOP 5 FASTEST (Training Time)")
    print("-" * 50)
    print_table_header()
    for metrics in sorted_metrics[:5]:
        print_result_row(metrics)
    
    # Memory efficiency rankings
    print(f"\n💾 TOP 5 MEMORY EFFICIENT")
    print("-" * 50)
    memory_sorted = sorted(all_metrics, key=lambda x: x['peak_memory'])
    print_table_header()
    for metrics in memory_sorted[:5]:
        print_result_row(metrics)
    
    # Accuracy rankings
    print(f"\n🎯 TOP 5 MOST ACCURATE")
    print("-" * 50)
    accuracy_sorted = sorted(all_metrics, key=lambda x: x['accuracy'], reverse=True)
    print_table_header()
    for metrics in accuracy_sorted[:5]:
        print_result_row(metrics)
    
    # Language comparison
    print(f"\n🐍🦀 PYTHON vs RUST SUMMARY")
    print("-" * 50)
    
    python_results = [m for m in all_metrics if m['language'] == 'Python']
    rust_results = [m for m in all_metrics if m['language'] == 'Rust']
    
    if python_results and rust_results:
        py_avg_time = sum(r['training_time'] for r in python_results) / len(python_results)
        rust_avg_time = sum(r['training_time'] for r in rust_results) / len(rust_results)
        
        py_avg_mem = sum(r['peak_memory'] for r in python_results) / len(python_results)
        rust_avg_mem = sum(r['peak_memory'] for r in rust_results) / len(rust_results)
        
        py_avg_acc = sum(r['accuracy'] for r in python_results) / len(python_results)
        rust_avg_acc = sum(r['accuracy'] for r in rust_results) / len(rust_results)
        
        print(f"Python ({len(python_results)} tests):")
        print(f"  Average Training Time: {py_avg_time:.3f}s")
        print(f"  Average Memory Usage:  {py_avg_mem:.1f} MB") 
        print(f"  Average Accuracy:      {py_avg_acc:.2f}%")
        
        print(f"\nRust ({len(rust_results)} tests):")
        print(f"  Average Training Time: {rust_avg_time:.3f}s")
        print(f"  Average Memory Usage:  {rust_avg_mem:.1f} MB")
        print(f"  Average Accuracy:      {rust_avg_acc:.2f}%")
        
        print(f"\nComparison:")
        if rust_avg_time < py_avg_time:
            speedup = py_avg_time / rust_avg_time
            print(f"  🚀 Rust is {speedup:.2f}x FASTER")
        else:
            slowdown = rust_avg_time / py_avg_time
            print(f"  🐌 Rust is {slowdown:.2f}x SLOWER")
            
        if rust_avg_mem < py_avg_mem:
            efficiency = py_avg_mem / rust_avg_mem
            print(f"  💾 Rust uses {efficiency:.2f}x LESS memory")
        else:
            waste = rust_avg_mem / py_avg_mem
            print(f"  💾 Rust uses {waste:.2f}x MORE memory")
            
        if rust_avg_acc > py_avg_acc:
            print(f"  🎯 Rust is {rust_avg_acc - py_avg_acc:.2f}% more accurate")
        else:
            print(f"  🎯 Python is {py_avg_acc - rust_avg_acc:.2f}% more accurate")
    
    # Dataset analysis
    datasets = set(m['dataset'] for m in all_metrics)
    print(f"\n📈 ANALYSIS BY DATASET")
    print("-" * 50)
    
    for dataset in sorted(datasets):
        dataset_results = [m for m in all_metrics if m['dataset'] == dataset]
        print(f"\n🔍 {dataset} Dataset ({len(dataset_results)} tests):")
        
        fastest = min(dataset_results, key=lambda x: x['training_time'])
        most_accurate = max(dataset_results, key=lambda x: x['accuracy'])
        
        print(f"  ⚡ Fastest: {fastest['language']} {fastest['architecture']} {fastest['device']} ({fastest['training_time']:.3f}s)")
        print(f"  🎯 Most Accurate: {most_accurate['language']} {most_accurate['architecture']} {most_accurate['device']} ({most_accurate['accuracy']:.2f}%)")

    # Save summary to file
    print(f"\n💾 Saving summary to benchmark_summary.txt...")
    with open("benchmark_summary.txt", "w") as f:
        f.write("BENCHMARK RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total benchmarks: {len(all_metrics)}\n")
        f.write(f"Python tests: {len(python_results)}\n") 
        f.write(f"Rust tests: {len(rust_results)}\n\n")
        
        if python_results and rust_results:
            f.write("AVERAGES:\n")
            f.write(f"Python - Time: {py_avg_time:.3f}s, Memory: {py_avg_mem:.1f}MB, Accuracy: {py_avg_acc:.2f}%\n")
            f.write(f"Rust   - Time: {rust_avg_time:.3f}s, Memory: {rust_avg_mem:.1f}MB, Accuracy: {rust_avg_acc:.2f}%\n")

def main():
    print("🔍 Simple Deep Learning Benchmark Analysis")
    print("=" * 50)
    
    # Load all results
    results = load_benchmark_results()
    
    if not results:
        print("❌ No benchmark results found!")
        print("💡 Make sure benchmark files exist in current directory")
        print("   Expected pattern: *_complete_training_results.json")
        return
    
    print(f"✅ Loaded {len(results)} benchmark results")
    
    # Analyze results
    analyze_results(results)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📄 Summary saved to: benchmark_summary.txt")

if __name__ == "__main__":
    main()