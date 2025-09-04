#!/usr/bin/env python3
"""
Fix Accuracy Display Script
Recalculates and displays correct accuracy values from existing benchmark results
"""

import json
import glob

def fix_accuracy_display(result):
    """Fix accuracy display based on the framework and values"""
    accuracy = result['quality_metrics']['accuracy']
    
    if accuracy is None:
        return 0.0
    
    # If accuracy is > 1, it's likely already in percentage format
    # If accuracy is <= 1, it's in decimal format (0.0-1.0)
    if accuracy > 1:
        return accuracy  # Already in percentage
    else:
        return accuracy * 100  # Convert decimal to percentage

def analyze_fixed_results():
    """Analyze results with corrected accuracy display"""
    
    # Load all result files
    pattern = "*_complete_training_results.json"
    files = glob.glob(pattern)
    
    print("🔧 CORRECTED ACCURACY ANALYSIS")
    print("=" * 80)
    print(f"📁 Analyzing {len(files)} result files...")
    print()
    
    results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    # Extract and fix metrics
    fixed_metrics = []
    
    print("📊 CORRECTED ACCURACY TABLE")
    print("-" * 100)
    print(f"{'Language':<8} {'Dataset':<12} {'Architecture':<12} {'Device':<6} {'Time(s)':<8} {'Accuracy(%)':<12} {'Loss':<8}")
    print("-" * 100)
    
    for result in results:
        try:
            # Extract basic info
            language = result['language'].title()
            dataset = result['dataset'].title()
            architecture = result['metadata']['architecture']
            device = result['metadata'].get('device', 'Unknown').replace('Cuda(0)', 'GPU').replace('Cpu', 'CPU')
            training_time = result['performance_metrics']['training_time_seconds']
            loss = result['quality_metrics']['loss']
            
            # Fix accuracy calculation
            corrected_accuracy = fix_accuracy_display(result)
            
            # Store corrected metrics
            fixed_metrics.append({
                'language': language,
                'dataset': dataset,
                'architecture': architecture,
                'device': device,
                'training_time': training_time,
                'accuracy': corrected_accuracy,
                'loss': loss,
                'run_id': result['run_id']
            })
            
            # Print row
            print(f"{language:<8} {dataset:<12} {architecture:<12} {device:<6} {training_time:<8.3f} {corrected_accuracy:<12.2f} {loss:<8.3f}")
            
        except KeyError as e:
            print(f"❌ Missing key {e} in result: {result.get('run_id', 'unknown')}")
    
    # Analysis with corrected accuracies
    print(f"\n🎯 ACCURACY RANKINGS (CORRECTED)")
    print("-" * 50)
    
    # Sort by corrected accuracy
    accuracy_sorted = sorted(fixed_metrics, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Language':<8} {'Dataset':<12} {'Architecture':<12} {'Device':<6} {'Accuracy(%)':<12}")
    print("-" * 70)
    
    for i, metrics in enumerate(accuracy_sorted[:10], 1):
        print(f"{i:<4} {metrics['language']:<8} {metrics['dataset']:<12} {metrics['architecture']:<12} "
              f"{metrics['device']:<6} {metrics['accuracy']:<12.2f}")
    
    # Language comparison with corrected accuracies
    print(f"\n🐍🦀 PYTHON vs RUST - CORRECTED COMPARISON")
    print("-" * 60)
    
    python_results = [m for m in fixed_metrics if m['language'] == 'Python']
    rust_results = [m for m in fixed_metrics if m['language'] == 'Rust']
    
    if python_results and rust_results:
        py_avg_acc = sum(r['accuracy'] for r in python_results) / len(python_results)
        rust_avg_acc = sum(r['accuracy'] for r in rust_results) / len(rust_results)
        
        py_avg_time = sum(r['training_time'] for r in python_results) / len(python_results)
        rust_avg_time = sum(r['training_time'] for r in rust_results) / len(rust_results)
        
        print(f"Python ({len(python_results)} tests):")
        print(f"  Average Accuracy:      {py_avg_acc:.2f}%")
        print(f"  Average Training Time: {py_avg_time:.3f}s")
        
        print(f"\nRust ({len(rust_results)} tests):")
        print(f"  Average Accuracy:      {rust_avg_acc:.2f}%")
        print(f"  Average Training Time: {rust_avg_time:.3f}s")
        
        print(f"\nCORRECTED COMPARISON:")
        if py_avg_acc > rust_avg_acc:
            acc_diff = py_avg_acc - rust_avg_acc
            print(f"  🎯 Python is {acc_diff:.2f}% MORE ACCURATE")
        else:
            acc_diff = rust_avg_acc - py_avg_acc
            print(f"  🎯 Rust is {acc_diff:.2f}% MORE ACCURATE")
            
        speed_ratio = py_avg_time / rust_avg_time
        print(f"  ⚡ Rust is {speed_ratio:.1f}x FASTER")
    
    # Dataset-specific corrected analysis
    print(f"\n📈 DATASET ANALYSIS (CORRECTED ACCURACIES)")
    print("-" * 60)
    
    datasets = set(m['dataset'] for m in fixed_metrics)
    for dataset in sorted(datasets):
        dataset_results = [m for m in fixed_metrics if m['dataset'] == dataset]
        print(f"\n🔍 {dataset} Dataset:")
        
        # Best accuracy for this dataset
        best_accuracy = max(dataset_results, key=lambda x: x['accuracy'])
        fastest = min(dataset_results, key=lambda x: x['training_time'])
        
        print(f"  🏆 Best Accuracy: {best_accuracy['language']} {best_accuracy['architecture']} {best_accuracy['device']} ({best_accuracy['accuracy']:.2f}%)")
        print(f"  ⚡ Fastest: {fastest['language']} {fastest['architecture']} {fastest['device']} ({fastest['training_time']:.3f}s)")
    
    print(f"\n✅ ACCURACY ANALYSIS COMPLETE!")
    print(f"🎯 The accuracy values should now be displayed correctly!")

if __name__ == "__main__":
    analyze_fixed_results()