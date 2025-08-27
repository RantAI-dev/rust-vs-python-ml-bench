Assessment and Quality Plan (as of now)

Current state vs. docs:
- Implementation is broadly comprehensive. Some inconsistencies exist between code and schema enums, a deprecated dataset import, and certain placeholders in Rust LLM/RL used for benchmarking simplicity.

Scores (10 = excellent):
- Python benchmarks: 8.8 — strong coverage and metrics; minor enum mismatches and deprecated dataset usage fixed now.
- Rust benchmarks: 7.9 — broad coverage; several components are synthetic/placeholder for data/models; quality metrics simplified in places.
- Nextflow orchestration: 8.5 — complete and structured; scripts referenced appear present; assumes container paths; good labels.
- Config and docs: 8.7 — extensive; some claims exceed actual fidelity (e.g., fully production LLM in Rust).
- Tests: 7.5 — cover structure and schemas; could add runtime smoke tests.
- Overall: 8.3

Plan to reach 9.5+ overall:
1) Python consistency and robustness (priority: done/short-term)
   - Align enum constants usage with `src/shared/schemas/metrics.py` (done).
   - Replace deprecated `load_boston` with supported alternative (done: fallback to diabetes).
   - Add smoke tests invoking one run per domain to validate JSON outputs and schema round-trip.

2) Rust benchmarks fidelity (priority: medium)
   - Replace synthetic dataset stubs with proper loaders or bundled CSVs where referenced.
   - Improve metric calculations (SVM/clustering percentiles, proper accuracy/F1 where feasible).
   - Add GPU resource capture hooks where available; otherwise document limitations.

3) LLM and RL realism (priority: medium)
   - For Rust LLM, document placeholder nature; optionally wire minimal real tokenization pipeline via candle/tokenizers if feasible.
   - For RL, factor shared env/utilities and add deterministic seeds and repeatability; add mean/std reporting parity with Python.

4) Nextflow and reproducibility (priority: medium)
   - Add a small end-to-end profile invoking one Python and one Rust benchmark and flowing through phases.
   - Ensure container path expectations match repo layout or provide local profile.

5) Testing and CI (priority: medium)
   - Add pytest-based smoke tests to execute minimal runs for each benchmark category and validate JSON schema.
   - Add Rust unit tests for metric helpers; cargo test skeletons per crate.

6) Documentation accuracy (priority: low)
   - Adjust README/SPECS claims to match realistic state where placeholders exist; or elevate code to match claims.

Execution tracker:
- [x] Fix enum mismatches and legacy aliases in `src/shared/schemas/metrics.py`.
- [x] Fix deprecated dataset in regression benchmark.
- [x] Smoke workflow green for CNN, LLM, RL, RNN.
- [x] Document venv setup and `-resume` flow for classical ML in `USERGUIDE.md`; link from `README.md` and `SPECS.md`.
- [ ] Add smoke tests for Python/Rust execution JSON outputs.
- [ ] Tighten Rust metrics and replace simplified percentiles.
- [ ] Add small Nextflow profile + local run instructions.

