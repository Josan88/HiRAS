---
name: hiras-research-reproduction
description: Use when reproducing experiments from research papers, generating code from academic papers, evaluating paper-to-code quality, debugging failed paper reproductions, or planning a structured replication workflow. Triggers on requests involving paper reproduction, paper-to-code generation, experiment replication, scientific code implementation, PaperBench/Paper2Code evaluation, or multi-agent research workflows. Even if the user does not name HiRAS, apply this skill when they want to go from a paper to working code, validate a generated repository, or systematically debug a reproduction pipeline.
---

# HiRAS Research Reproduction

A methodology for structured, multi-phase experiment reproduction from research papers. Based on the Hierarchical Research Agent System (HiRAS) framework (Hong et al., 2025). Works in any coding workspace — not tied to a specific repository or toolchain.

## When to use

- Reproducing experiments from a research paper (ML, CS, scientific computing)
- Generating a codebase from a paper's methodology
- Evaluating the quality of a generated paper-to-code repository
- Debugging a failed reproduction where code was generated but does not execute
- Planning a structured, verifiable replication workflow for a published method
- User mentions: paper reproduction, paper-to-code, experiment replication, replicate this paper, reproduce the results, implement this paper, code from paper

## Core principle

Research reproduction fails most often because early-stage errors propagate unchecked. The HiRAS methodology addresses this with hierarchical supervision: a manager agent inspects intermediate artefacts at each phase boundary, validates completeness and correctness, and only then proceeds. When execution fails, the manager backtracks to the appropriate earlier phase rather than patching symptoms.

## The HiRAS workflow

Execute these phases in order. After each phase, validate the artefacts before moving to the next. If validation fails, loop back to the responsible phase.

### Phase 0: Intake

Read the paper completely. Do not skim, do not read only the abstract. Identify:

- The core research question and claimed contributions
- The experimental setup: datasets, models, baselines, metrics
- Hyperparameters and training details (learning rate, batch size, epochs, optimizer, etc.)
- Any addendum, supplementary materials, or linked repositories
- What is explicitly stated vs. what would need to be inferred

**Artefact:** A structured intake summary (can be inline notes or a file). Flag any ambiguities or missing information explicitly.

### Phase 1: Overall planning (`plan.md`)

Produce a high-level implementation roadmap covering every experiment in the paper. For each experiment:

- Goal and expected outcome
- Required datasets and where to obtain them
- Model architecture or algorithm to implement
- Evaluation protocol and metrics
- Dependencies on other experiments

Do not go into function-level detail yet. Stay at the experiment-section level.

**Gate:** The plan must cover every experiment described in the paper. If the paper has N experiment sections, the plan must address all N.

### Phase 2: Architecture design (`architecture.md`)

Specify the software structure needed to implement the plan:

- Directory layout
- Module/file list with responsibilities
- Data structures and interfaces between modules
- Entry point(s) and execution flow
- External dependencies (libraries, frameworks)

Follow the paper's conceptual structure. If the paper describes a pipeline with stages X, Y, Z, your architecture should have corresponding modules.

**Gate:** Every experiment from `plan.md` must be traceable to one or more modules in the architecture.

### Phase 3: Dependency modelling (`dependency.md`)

For each module identified in the architecture:

- Required packages with versions (requirements.txt format)
- Inter-file dependencies (which module imports which)
- Logic analysis: classes, methods, functions to implement, with descriptions
- Task ordering: which files must be implemented first based on dependency order

**Gate:** The task list must be topologically sorted. No circular imports.

### Phase 4: Configuration generation (`config.yaml`)

Extract all hyperparameters and experimental parameters from the paper. Do NOT fabricate values.

- Learning rate, batch size, epochs, optimizer settings
- Model architecture dimensions
- Dataset paths and preprocessing settings
- Evaluation settings (metrics, splits, seeds)
- Any hardware/environment assumptions

For values not explicitly stated in the paper, mark them as `UNKNOWN` or `TODO` with a note about what the paper does or does not specify.

**Gate:** Every hyperparameter mentioned in the paper must appear in this file. Values not from the paper must be clearly flagged.

### Phase 5: Implementation analysis (`analysis/`)

Before writing any code, analyse each component:

1. Create `analysis/components.txt` listing every file to implement
2. For each component, create `analysis/<component_name>_analysis.md` with:
   - Introduction: what this component does
   - Implementation: step-by-step logic, referencing the paper's methodology
   - Notes: interactions with other components, edge cases

Read `plan.md`, `architecture.md`, `dependency.md`, and `config.yaml` before analysing each component.

**Gate:** `analysis/components.txt` must list all files from the architecture. Every listed component must have a corresponding analysis file.

### Phase 6: Code generation

Implement one component at a time. For each:

1. Read the relevant analysis file
2. Read any dependency code already written
3. Implement the component following the analysis and architecture
4. Verify imports resolve and basic syntax is correct

Follow the architecture design exactly. Do not add public interfaces not in the design. Do not leave TODOs in the code — implement everything or explicitly raise an issue.

**Gate:** Every component in `analysis/components.txt` must have a corresponding implementation file. No placeholder code.

### Phase 7: Execution and validation

Execute the code to reproduce the paper's experiments:

1. Set up the execution environment (install dependencies, prepare data)
2. Run each experiment as specified in the plan
3. Record actual results — do NOT fabricate or approximate
4. Compare results against the paper's reported numbers where possible
5. Log all execution output, errors, and warnings

**Gate:** All experiments from the plan must have been attempted. All results must be actual execution outputs.

### Phase 8: Report

Produce a reproduction report covering:

- Which experiments succeeded and which failed
- Observed results vs. paper's reported results
- Explanation of any discrepancies
- Remaining issues and potential fixes

## Backtracking rules

When execution fails, do not just fix the symptom. Diagnose the root cause:

| Failure type | Backtrack to |
|---|---|
| Import/module not found | Phase 3 (dependency) or Phase 6 (code) |
| Wrong results (code runs but output incorrect) | Phase 5 (analysis) or Phase 6 (code) |
| Missing experiment not attempted | Phase 1 (plan) |
| Missing hyperparameter or wrong config | Phase 4 (config) |
| Runtime error in code | Phase 6 (code), then re-execute Phase 7 |
| Paper ambiguity causing incorrect assumptions | Phase 0 (intake), document the ambiguity |

## Evaluation mode (P2C-Ex)

When asked to evaluate a generated paper-to-code repository:

1. Read the paper completely
2. Read the generated repository: file structure + all code files
3. Score on a 1-5 scale for correctness:
   - 1: Does not implement core concepts
   - 2: Attempts but significant gaps/errors
   - 3: Some core components correct, notable errors
   - 4: Key components correct, minor inaccuracies only
   - 5: Fully and accurately implements all key components
4. Provide structured critique with severity levels (high/medium/low)
5. Output as JSON:

```json
{
  "critique_list": [
    {
      "file_name": "example.py",
      "func_name": "train_loop",
      "severity_level": "high",
      "critique": "Description of the issue."
    }
  ],
  "score": 3
}
```

To reduce evaluator hallucination (a known problem with reference-free evaluation), always:

- Count the total files in the repository
- List the hierarchical file structure
- Verify that documented functionality actually exists as implemented code, not just README descriptions

## Evidence discipline

These rules prevent the most common failure modes in automated reproduction:

1. **Never fabricate results.** Only report execution outputs you actually observed.
2. **Never invent hyperparameters.** If the paper doesn't specify a value, say so.
3. **Distinguish paper-stated from inferred.** When you must make assumptions, label them explicitly.
4. **Read completely before acting.** Every phase must read all prior artefacts fully, not skim.
5. **One component at a time.** Do not batch-generate multiple files without checking each one.
6. **Manager checkpoints exist for a reason.** Never skip phase validation to "save time."

## Adapting to your workspace

This skill is methodology-based, not tool-based. Adapt to whatever you have:

- **No conda/virtualenv:** Use whatever environment management is available, or note the limitation.
- **No GPU:** Note which experiments require GPU and cannot be run.
- **Partial paper:** Work with what you have, flag what's missing.
- **No execution environment:** Focus on Phases 0-6 (planning through code generation) and document what would need to be executed.
- **Single-file paper:** Adapt the phase granularity. A simple paper may not need separate dependency modelling and architecture design files — inline the key decisions.

The methodology scales. A toy 2-page paper might need 10 minutes of planning. A 30-page ML paper with 8 experiments might need extensive planning, architecture, and dependency analysis before any code is written.

## Reference files

For detailed guidance, read:

- `references/hiras-methodology.md` — HiRAS framework details from the original paper
- `references/reproduction-artifacts.md` — artefact specifications and quality gates per phase
- `references/p2c-ex-evaluation.md` — P2C-Ex evaluation protocol and prompt template
- `references/repo-usage-notes.md` — notes for running the original HiRAS implementation from the GitHub repo
