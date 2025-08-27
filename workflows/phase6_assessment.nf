#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE6_ASSESSMENT {
    take:
        statistical_results
        benchmark_results
    
    main:
        // Assess ecosystem maturity
        ASSESS_ECOSYSTEM_MATURITY(benchmark_results)
        
        // Evaluate framework capabilities
        EVALUATE_FRAMEWORK_CAPABILITIES(benchmark_results)
        
        // Generate recommendations
        GENERATE_RECOMMENDATIONS(
            ASSESS_ECOSYSTEM_MATURITY.out.ecosystem_assessment,
            EVALUATE_FRAMEWORK_CAPABILITIES.out.framework_evaluation,
            statistical_results
        )
        
        // Generate final comprehensive report
        GENERATE_FINAL_REPORT(
            statistical_results,
            ASSESS_ECOSYSTEM_MATURITY.out.ecosystem_assessment,
            EVALUATE_FRAMEWORK_CAPABILITIES.out.framework_evaluation,
            GENERATE_RECOMMENDATIONS.out.recommendations
        )
    
    emit:
        final_report = GENERATE_FINAL_REPORT.out.final_report
        recommendations = GENERATE_RECOMMENDATIONS.out.recommendations
}

process ASSESS_ECOSYSTEM_MATURITY {
    tag "Ecosystem Maturity Assessment"
    label 'cpu_intensive'
    
    input:
    path benchmark_results
    
    output:
    path "ecosystem_assessment.json", emit: ecosystem_assessment
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/assess_ecosystem_maturity.py \
        --benchmark-results ${benchmark_results} \
        --output ecosystem_assessment.json
    """
}

process EVALUATE_FRAMEWORK_CAPABILITIES {
    tag "Framework Capabilities Evaluation"
    label 'cpu_intensive'
    
    input:
    path benchmark_results
    
    output:
    path "framework_evaluation.json", emit: framework_evaluation
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/evaluate_framework_capabilities.py \
        --benchmark-results ${benchmark_results} \
        --output framework_evaluation.json
    """
}

process GENERATE_RECOMMENDATIONS {
    tag "Recommendations Generation"
    label 'cpu_intensive'
    
    input:
    path ecosystem_assessment
    path framework_evaluation
    path statistical_results
    
    output:
    path "recommendations.json", emit: recommendations
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/generate_recommendations.py \
        --ecosystem-assessment ${ecosystem_assessment} \
        --framework-evaluation ${framework_evaluation} \
        --statistical-results ${statistical_results} \
        --output recommendations.json
    """
}

process GENERATE_FINAL_REPORT {
    tag "Final Report Generation"
    label 'cpu_intensive'
    
    input:
    path statistical_results
    path ecosystem_assessment
    path framework_evaluation
    path recommendations
    
    output:
    path "final_comprehensive_report.md", emit: final_report
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/generate_final_report.py \
        --statistical-results ${statistical_results} \
        --ecosystem-assessment ${ecosystem_assessment} \
        --framework-evaluation ${framework_evaluation} \
        --recommendations ${recommendations} \
        --output final_comprehensive_report.md
    """
} 