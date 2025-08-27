#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE5_ANALYSIS {
    take:
        benchmark_results
    
    main:
        // Perform statistical analysis
        PERFORM_STATISTICAL_ANALYSIS(benchmark_results)
        
        // Generate performance comparisons
        GENERATE_PERFORMANCE_COMPARISONS(PERFORM_STATISTICAL_ANALYSIS.out.statistical_results)
        
        // Create visualizations
        CREATE_VISUALIZATIONS(PERFORM_STATISTICAL_ANALYSIS.out.statistical_results)
        
        // Generate analysis report
        GENERATE_ANALYSIS_REPORT(
            PERFORM_STATISTICAL_ANALYSIS.out.statistical_results,
            GENERATE_PERFORMANCE_COMPARISONS.out.comparisons,
            CREATE_VISUALIZATIONS.out.visualizations
        )
    
    emit:
        statistical_results = PERFORM_STATISTICAL_ANALYSIS.out.statistical_results
        analysis_report = GENERATE_ANALYSIS_REPORT.out.report
}

process PERFORM_STATISTICAL_ANALYSIS {
    tag "Statistical Analysis"
    label 'cpu_intensive'
    
    input:
    path benchmark_results
    
    output:
    path "statistical_results.json", emit: statistical_results
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/perform_statistical_analysis.py \
        --results ${benchmark_results} \
        --output statistical_results.json \
        --confidence-level 0.95
    """
}

process GENERATE_PERFORMANCE_COMPARISONS {
    tag "Performance Comparisons"
    label 'cpu_intensive'
    
    input:
    path statistical_results
    
    output:
    path "performance_comparisons.json", emit: comparisons
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/generate_performance_comparisons.py \
        --statistical-results ${statistical_results} \
        --output performance_comparisons.json
    """
}

process CREATE_VISUALIZATIONS {
    tag "Visualization Creation"
    label 'cpu_intensive'
    
    input:
    path statistical_results
    
    output:
    path "visualizations.json", emit: visualizations
    path "plots/", emit: plots
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/create_visualizations.py \
        --statistical-results ${statistical_results} \
        --output-visualizations visualizations.json \
        --output-plots plots/
    """
}

process GENERATE_ANALYSIS_REPORT {
    tag "Analysis Report Generation"
    label 'cpu_intensive'
    
    input:
    path statistical_results
    path performance_comparisons
    path visualizations
    
    output:
    path "analysis_report.md", emit: report
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/generate_analysis_report.py \
        --statistical-results ${statistical_results} \
        --performance-comparisons ${performance_comparisons} \
        --visualizations ${visualizations} \
        --output analysis_report.md
    """
} 