#!/usr/bin/env python3
"""
Benchmark Results Analysis Script
Analyzes all *_complete_training_results.json files and generates comparison tables
"""

import json
import glob
import pandas as pd
from pathlib import Path

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

def create_comparison_table(results):
    """Create comprehensive comparison table"""
    
    rows = []
    
    for result in results:
        # Extract key metrics
        row = {
            'Language': result['language'].title(),
            'Dataset': result['dataset'].title(),
            'Architecture': result['metadata']['architecture'],
            'Device': result['metadata'].get('device', 'Unknown').replace('Cuda(0)', 'GPU').replace('Cpu', 'CPU'),
            'Training_Time_s': result['performance_metrics']['training_time_seconds'],
            'Peak_Memory_MB': result['resource_metrics']['peak_memory_mb'],
            'Accuracy_%': result['quality_metrics']['accuracy'] * 100 if result['quality_metrics']['accuracy'] else 0,
            'GPU_Memory_MB': result['resource_metrics'].get('peak_gpu_memory_mb', 0) or 0,
            'CPU_Usage_%': result['resource_metrics']['cpu_utilization_percent'],
            'Final_Loss': result['quality_metrics']['loss'],
            'Run_ID': result['run_id']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def generate_summary_tables(df):
    """Generate various summary tables"""
    
    print("\n🏆 COMPREHENSIVE BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Overall performance table
    print(f"\n📊 COMPLETE RESULTS TABLE ({len(df)} benchmarks)")
    print("-" * 80)
    
    # Sort by training time for performance ranking
    df_sorted = df.sort_values('Training_Time_s')
    
    # Display table with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    print(df_sorted.to_string(index=False, float_format='%.3f'))
    
    # Performance analysis by dataset/architecture
    print("\n🚀 SPEED RANKINGS (Training Time)")
    print("-" * 50)
    speed_ranking = df_sorted[['Language', 'Dataset', 'Architecture', 'Device', 'Training_Time_s']]
    print(speed_ranking.head(10).to_string(index=False, float_format='%.3f'))
    
    # Memory efficiency rankings
    print("\n💾 MEMORY EFFICIENCY RANKINGS")
    print("-" * 50)
    memory_ranking = df.sort_values('Peak_Memory_MB')[['Language', 'Dataset', 'Architecture', 'Device', 'Peak_Memory_MB']]
    print(memory_ranking.head(10).to_string(index=False, float_format='%.1f'))
    
    # Accuracy rankings
    print("\n🎯 ACCURACY RANKINGS")
    print("-" * 50)
    accuracy_ranking = df.sort_values('Accuracy_%', ascending=False)[['Language', 'Dataset', 'Architecture', 'Device', 'Accuracy_%']]
    print(accuracy_ranking.head(10).to_string(index=False, float_format='%.2f'))
    
    # Dataset-specific analysis
    print("\n📈 ANALYSIS BY DATASET")
    print("-" * 50)
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        print(f"\n🔍 {dataset} Dataset:")
        print(dataset_df[['Language', 'Architecture', 'Device', 'Training_Time_s', 'Accuracy_%']].to_string(index=False, float_format='%.3f'))
    
    # Language comparison
    print("\n🐍🦀 PYTHON vs RUST COMPARISON")
    print("-" * 50)
    
    python_avg = df[df['Language'] == 'Python']['Training_Time_s'].mean()
    rust_avg = df[df['Language'] == 'Rust']['Training_Time_s'].mean()
    
    print(f"Average Training Time:")
    print(f"  Python: {python_avg:.3f}s")
    print(f"  Rust:   {rust_avg:.3f}s")
    print(f"  Speedup: {python_avg/rust_avg:.2f}x (Rust advantage)")
    
    python_mem = df[df['Language'] == 'Python']['Peak_Memory_MB'].mean()
    rust_mem = df[df['Language'] == 'Rust']['Peak_Memory_MB'].mean()
    
    print(f"\nAverage Peak Memory:")
    print(f"  Python: {python_mem:.1f} MB")
    print(f"  Rust:   {rust_mem:.1f} MB")
    print(f"  Efficiency: {python_mem/rust_mem:.2f}x (Python uses more)")
    
    return df

def save_csv_export(df):
    """Save results as CSV for further analysis"""
    csv_file = "benchmark_results_complete.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n📄 Results exported to: {csv_file}")

def main():
    print("🔍 Deep Learning Benchmark Results Analysis")
    print("=" * 50)
    
    # Load all results
    results = load_benchmark_results()
    
    if not results:
        print("❌ No benchmark results found!")
        print("💡 Make sure to run ./run_complete_benchmark.sh first")
        return
    
    print(f"✅ Loaded {len(results)} benchmark results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Generate summary
    generate_summary_tables(df)
    
    # Export CSV
    save_csv_export(df)
    
    print("\n🎉 Analysis complete!")
    print("\n💡 Key insights:")
    print("   - Check speed rankings for performance winners")
    print("   - Review memory rankings for efficiency")  
    print("   - Compare accuracy across platforms")
    print("   - Analyze dataset-specific patterns")

if __name__ == "__main__":
    main()