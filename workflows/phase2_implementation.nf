#!/usr/bin/env nextflow

nextflow.enable.dsl=2

workflow PHASE2_IMPLEMENTATION {
    take:
        selected_frameworks
    
    main:
        // Implement Python benchmarks
        IMPLEMENT_PYTHON_BENCHMARKS(selected_frameworks)
        
        // Implement Rust benchmarks
        IMPLEMENT_RUST_BENCHMARKS(selected_frameworks)
        
        // Validate implementations
        VALIDATE_IMPLEMENTATIONS(
            IMPLEMENT_PYTHON_BENCHMARKS.out.implementations,
            IMPLEMENT_RUST_BENCHMARKS.out.implementations
        )
        
        // Generate implementation report
        GENERATE_IMPLEMENTATION_REPORT(VALIDATE_IMPLEMENTATIONS.out.validated_implementations)
    
    emit:
        implementations = VALIDATE_IMPLEMENTATIONS.out.validated_implementations
        implementation_report = GENERATE_IMPLEMENTATION_REPORT.out.report
}

process IMPLEMENT_PYTHON_BENCHMARKS {
    tag "Python Implementation"
    label 'cpu_intensive'
    
    input:
    path selected_frameworks
    
    output:
    path "python_implementations.json", emit: implementations
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/implement_python_benchmarks.py \
        --frameworks ${selected_frameworks} \
        --output python_implementations.json \
        --language python
    """
}

process IMPLEMENT_RUST_BENCHMARKS {
    tag "Rust Implementation"
    label 'cpu_intensive'
    
    input:
    path selected_frameworks
    
    output:
    path "rust_implementations.json", emit: implementations
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/implement_rust_benchmarks.py \
        --frameworks ${selected_frameworks} \
        --output rust_implementations.json \
        --language rust
    """
}

process VALIDATE_IMPLEMENTATIONS {
    tag "Implementation Validation"
    label 'cpu_intensive'
    
    input:
    path python_implementations
    path rust_implementations
    
    output:
    path "validated_implementations.json", emit: validated_implementations
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/validate_implementations.py \
        --python ${python_implementations} \
        --rust ${rust_implementations} \
        --output validated_implementations.json
    """
}

process GENERATE_IMPLEMENTATION_REPORT {
    tag "Implementation Report Generation"
    label 'cpu_intensive'
    
    input:
    path validated_implementations
    
    output:
    path "implementation_report.md", emit: report
    
    script:
    """
    set -euo pipefail
    export PYTHONPATH="$launchDir"
    VENV="$launchDir/.venv"; PY="\$VENV/bin/python"; if [ ! -x "\$PY" ]; then echo "Missing venv at \$VENV" >&2; exit 1; fi
    "\$PY" "$launchDir"/scripts/generate_implementation_report.py \
        --implementations ${validated_implementations} \
        --output implementation_report.md
    """
} 