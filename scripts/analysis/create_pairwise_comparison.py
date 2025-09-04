#!/usr/bin/env python3
"""
Pairwise Python vs Rust Comparison
Creates head-to-head comparisons for same dataset+architecture combinations
"""

import json
import glob

def load_and_organize_results():
    """Load results and organize by dataset+architecture pairs"""
    
    # Load all result files
    pattern = "*_complete_training_results.json"
    files = glob.glob(pattern)
    
    results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    # Organize by dataset+architecture combinations
    pairs = {}
    
    for result in results:
        try:
            language = result['language'].title()
            dataset = result['dataset'].title()
            architecture = result['metadata']['architecture']
            device = result['metadata'].get('device', 'Unknown').replace('Cuda(0)', 'GPU').replace('Cpu', 'CPU')
            
            # Fix accuracy calculation
            accuracy = result['quality_metrics']['accuracy']
            if accuracy is None:
                accuracy = 0.0
            elif accuracy <= 1:
                accuracy = accuracy * 100  # Convert decimal to percentage
            
            key = f"{dataset}_{architecture}"
            
            if key not in pairs:
                pairs[key] = {'python': [], 'rust': []}
            
            result_data = {
                'language': language,
                'dataset': dataset,
                'architecture': architecture,
                'device': device,
                'training_time': result['performance_metrics']['training_time_seconds'],
                'accuracy': accuracy,
                'loss': result['quality_metrics']['loss'],
                'peak_memory': result['resource_metrics']['peak_memory_mb'],
                'gpu_memory': result['resource_metrics'].get('peak_gpu_memory_mb', 0) or 0,
                'run_id': result['run_id']
            }
            
            if language.lower() == 'python':
                pairs[key]['python'].append(result_data)
            else:
                pairs[key]['rust'].append(result_data)
                
        except KeyError as e:
            print(f"❌ Missing key {e} in result")
    
    return pairs

def print_pairwise_comparison(pairs):
    """Print detailed pairwise comparisons"""
    
    print("🥊 HEAD-TO-HEAD: PYTHON vs RUST PAIRWISE COMPARISON")
    print("=" * 100)
    print()
    
    comparison_summary = []
    
    for key, data in pairs.items():
        python_results = data['python']
        rust_results = data['rust']
        
        if not python_results or not rust_results:
            continue  # Skip if we don't have both languages
            
        dataset, architecture = key.split('_', 1)
        
        print(f"🔥 {dataset} + {architecture.upper()}")
        print("-" * 80)
        
        # Show all combinations
        for py_result in python_results:
            for rust_result in rust_results:
                print(f"📊 MATCHUP:")
                print(f"  🐍 Python {py_result['device']:<6}: {py_result['training_time']:<8.3f}s | {py_result['accuracy']:<6.2f}% | {py_result['peak_memory']:<8.1f}MB | GPU: {py_result['gpu_memory']:<6.1f}MB")
                print(f"  🦀 Rust   {rust_result['device']:<6}: {rust_result['training_time']:<8.3f}s | {rust_result['accuracy']:<6.2f}% | {rust_result['peak_memory']:<8.1f}MB | GPU: {rust_result['gpu_memory']:<6.1f}MB")
                
                # Calculate advantages
                speed_advantage = py_result['training_time'] / rust_result['training_time']
                accuracy_diff = py_result['accuracy'] - rust_result['accuracy']
                memory_ratio = py_result['peak_memory'] / rust_result['peak_memory']
                
                print(f"  📈 ADVANTAGE:")
                if speed_advantage > 1:
                    print(f"     ⚡ Rust is {speed_advantage:.1f}x FASTER")
                else:
                    print(f"     ⚡ Python is {1/speed_advantage:.1f}x FASTER")
                    
                if accuracy_diff > 0:
                    print(f"     🎯 Python is {accuracy_diff:.2f}% MORE ACCURATE")
                else:
                    print(f"     🎯 Rust is {abs(accuracy_diff):.2f}% MORE ACCURATE")
                    
                if memory_ratio > 1:
                    print(f"     💾 Rust uses {memory_ratio:.1f}x LESS memory")
                else:
                    print(f"     💾 Python uses {1/memory_ratio:.1f}x LESS memory")
                
                # Store for summary
                comparison_summary.append({
                    'dataset_arch': key,
                    'python_device': py_result['device'],
                    'rust_device': rust_result['device'],
                    'speed_advantage': speed_advantage,
                    'accuracy_diff': accuracy_diff,
                    'memory_ratio': memory_ratio,
                    'python_time': py_result['training_time'],
                    'rust_time': rust_result['training_time'],
                    'python_accuracy': py_result['accuracy'],
                    'rust_accuracy': rust_result['accuracy']
                })
                
                print()
        
        print()
    
    return comparison_summary

def print_summary_table(comparison_summary):
    """Print a clean summary table"""
    
    print("📋 PAIRWISE COMPARISON SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Dataset+Architecture':<25} {'Devices':<12} {'Speed Winner':<15} {'Accuracy Winner':<18} {'Memory Winner':<15}")
    print("-" * 120)
    
    for comp in comparison_summary:
        dataset_arch = comp['dataset_arch'].replace('_', ' + ')
        devices = f"{comp['python_device'][:3]}vs{comp['rust_device'][:3]}"
        
        # Determine winners
        if comp['speed_advantage'] > 1:
            speed_winner = f"Rust ({comp['speed_advantage']:.1f}x)"
        else:
            speed_winner = f"Python ({1/comp['speed_advantage']:.1f}x)"
            
        if comp['accuracy_diff'] > 0:
            acc_winner = f"Python (+{comp['accuracy_diff']:.1f}%)"
        else:
            acc_winner = f"Rust (+{abs(comp['accuracy_diff']):.1f}%)"
            
        if comp['memory_ratio'] > 1:
            mem_winner = f"Rust ({comp['memory_ratio']:.1f}x less)"
        else:
            mem_winner = f"Python ({1/comp['memory_ratio']:.1f}x less)"
        
        print(f"{dataset_arch:<25} {devices:<12} {speed_winner:<15} {acc_winner:<18} {mem_winner:<15}")

def print_overall_stats(comparison_summary):
    """Print overall statistics"""
    
    print(f"\n🏆 OVERALL PAIRWISE STATISTICS ({len(comparison_summary)} matchups)")
    print("-" * 60)
    
    rust_speed_wins = sum(1 for c in comparison_summary if c['speed_advantage'] > 1)
    python_acc_wins = sum(1 for c in comparison_summary if c['accuracy_diff'] > 0)
    rust_memory_wins = sum(1 for c in comparison_summary if c['memory_ratio'] > 1)
    
    print(f"⚡ Speed Winners:")
    print(f"  Rust:   {rust_speed_wins}/{len(comparison_summary)} ({rust_speed_wins/len(comparison_summary)*100:.1f}%)")
    print(f"  Python: {len(comparison_summary)-rust_speed_wins}/{len(comparison_summary)} ({(len(comparison_summary)-rust_speed_wins)/len(comparison_summary)*100:.1f}%)")
    
    print(f"\n🎯 Accuracy Winners:")
    print(f"  Python: {python_acc_wins}/{len(comparison_summary)} ({python_acc_wins/len(comparison_summary)*100:.1f}%)")
    print(f"  Rust:   {len(comparison_summary)-python_acc_wins}/{len(comparison_summary)} ({(len(comparison_summary)-python_acc_wins)/len(comparison_summary)*100:.1f}%)")
    
    print(f"\n💾 Memory Winners:")
    print(f"  Rust:   {rust_memory_wins}/{len(comparison_summary)} ({rust_memory_wins/len(comparison_summary)*100:.1f}%)")
    print(f"  Python: {len(comparison_summary)-rust_memory_wins}/{len(comparison_summary)} ({(len(comparison_summary)-rust_memory_wins)/len(comparison_summary)*100:.1f}%)")
    
    # Average advantages
    avg_speed_advantage = sum(c['speed_advantage'] for c in comparison_summary) / len(comparison_summary)
    avg_accuracy_diff = sum(c['accuracy_diff'] for c in comparison_summary) / len(comparison_summary)
    avg_memory_ratio = sum(c['memory_ratio'] for c in comparison_summary) / len(comparison_summary)
    
    print(f"\n📊 AVERAGE ADVANTAGES:")
    print(f"  ⚡ Rust is {avg_speed_advantage:.1f}x faster on average")
    print(f"  🎯 Python is {avg_accuracy_diff:.1f}% more accurate on average")
    print(f"  💾 Rust uses {avg_memory_ratio:.1f}x less memory on average")

def main():
    print("🥊 Creating Pairwise Python vs Rust Comparison")
    print("=" * 50)
    
    # Load and organize results
    pairs = load_and_organize_results()
    
    if not pairs:
        print("❌ No matching pairs found!")
        return
    
    print(f"✅ Found {len(pairs)} dataset+architecture combinations")
    print()
    
    # Print detailed comparisons
    comparison_summary = print_pairwise_comparison(pairs)
    
    # Print summary table
    print_summary_table(comparison_summary)
    
    # Print overall statistics
    print_overall_stats(comparison_summary)
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   - Rust dominates in SPEED across almost all matchups")
    print(f"   - Python excels in MODEL ACCURACY (but Rust models aren't training properly)")
    print(f"   - Memory usage varies by combination")
    print(f"   - The accuracy gap shows Rust training bug needs fixing!")

if __name__ == "__main__":
    main()