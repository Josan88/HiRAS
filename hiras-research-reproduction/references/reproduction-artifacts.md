# Reproduction Artefacts Reference

## Artefact specification

### plan.md — Overall experiment plan

**Purpose:** High-level roadmap covering every experiment in the paper.

**Must contain:**
- One section per experiment from the paper
- For each experiment: goal, datasets, model/algorithm, evaluation protocol, metrics
- Dependencies between experiments
- Expected outcomes (what the paper claims)

**Must NOT contain:**
- Function-level implementation details (that's for analysis)
- Fabricated experiments not described in the paper

**Validation checklist:**
- [ ] Covers every experiment section in the paper
- [ ] Each experiment has a clear goal
- [ ] Datasets and metrics are paper-sourced
- [ ] Dependencies between experiments are identified

---

### architecture.md — Software structure design

**Purpose:** Module/file layout mapping the paper's methodology to code.

**Must contain:**
- Directory structure
- File list with responsibilities (one sentence each)
- Interfaces between modules (what each module exposes/consumes)
- Entry point and execution flow
- External library dependencies

**Must NOT contain:**
- Actual code implementations
- Module designs not traceable to the plan

**Validation checklist:**
- [ ] Every experiment in `plan.md` maps to one or more modules
- [ ] No orphan modules (every module serves at least one experiment)
- [ ] Interfaces are specified (not just "module A talks to module B")
- [ ] Entry point is clearly identified

---

### dependency.md — Dependency modelling

**Purpose:** Package requirements, inter-file dependencies, and task ordering.

**Must contain:**
- Required packages in requirements.txt format
- Logic analysis per file: classes, methods, functions with descriptions
- Inter-file import relationships
- Task list ordered by dependency (topological sort)

**Must NOT contain:**
- Packages not actually needed by the implementation
- Circular dependency chains

**Validation checklist:**
- [ ] All external packages from `architecture.md` are listed
- [ ] Task list is topologically sorted (no forward dependencies)
- [ ] Each file's key classes/functions are described
- [ ] No circular imports

---

### config.yaml — Configuration

**Purpose:** All hyperparameters and experimental parameters.

**Must contain:**
- Every hyperparameter mentioned in the paper (learning rate, batch size, epochs, etc.)
- Model architecture dimensions
- Dataset paths and preprocessing settings
- Evaluation settings
- Clear marking of values not from the paper: `# INFERRED: paper does not specify`

**Must NOT contain:**
- Fabricated values presented as paper-specified
- Hardcoded paths that assume a specific filesystem layout (use relative paths or variables)

**Validation checklist:**
- [ ] Every hyperparameter mentioned in the paper appears here
- [ ] Values not from the paper are explicitly flagged
- [ ] No fabricated values presented as fact

---

### analysis/ — Implementation analysis

**Purpose:** Per-component implementation guidance before coding begins.

**Files:**
- `components.txt` — list of all files to implement, one per line
- `<component_name>_analysis.md` — per-component analysis

**Each analysis file must contain:**
- Introduction: what this component does and why
- Implementation: step-by-step logic, referencing the paper
- Notes: interactions with other components, edge cases, caveats

**Validation checklist:**
- [ ] `components.txt` lists all files from architecture
- [ ] Every component in `components.txt` has a corresponding analysis file
- [ ] Each analysis references specific sections/methods from the paper
- [ ] Implementation steps are actionable (not vague "implement the model")

---

### code/ — Implementation

**Purpose:** Working code implementing the paper's methodology.

**Must contain:**
- One file per component listed in `analysis/components.txt`
- Complete implementations (no TODOs, no placeholder code)
- Imports that resolve to actual dependencies

**Must NOT contain:**
- Code for components not in the architecture
- Dead code or unused functions
- Hardcoded paths that assume a specific machine

**Validation checklist:**
- [ ] Every component in `components.txt` has a corresponding code file
- [ ] No TODO or placeholder comments
- [ ] All imports resolve
- [ ] Code follows the architecture design (interfaces match)

---

### results/ — Execution output

**Purpose:** Actual execution logs and results.

**Must contain:**
- Log files for each experiment attempted
- Actual observed outputs (metrics, values, plots)
- Error messages if experiments failed

**Must NOT contain:**
- Fabricated results
- Results from a different run presented as this run's output

**Validation checklist:**
- [ ] Every experiment from the plan has been attempted
- [ ] Results are actual execution outputs
- [ ] Failed experiments have error logs
- [ ] Results can be compared against paper's claims

---

## Phase quality gates summary

| Phase | Gate condition | Failure action |
|---|---|---|
| 0: Intake | Paper read completely, ambiguities flagged | Stay in Phase 0 |
| 1: Planning | All experiments covered | Re-do Phase 1 |
| 2: Architecture | All experiments map to modules | Re-do Phase 2 |
| 3: Dependency | Topologically sorted, no cycles | Re-do Phase 3 |
| 4: Config | All paper hyperparameters present | Re-do Phase 4 |
| 5: Analysis | All components analysed | Re-do Phase 5 |
| 6: Code | All components implemented, no TODOs | Re-do Phase 6 |
| 7: Execution | All experiments attempted, results recorded | Diagnose root cause, backtrack |
