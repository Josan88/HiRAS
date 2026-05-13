# HiRAS Backtracking Debug Plan

## Error Summary
```
ImportError: cannot import name 'GraphNet' from 'models.gnn'
```

---

## Root Cause Diagnosis

### Phase Attribution

| Phase | Artefact | Status |
|-------|----------|--------|
| Phase 0 (Intake) | Paper summary | ✓ Presumably correct |
| Phase 1 (Plan) | `plan.md` | ✓ Presumably correct |
| **Phase 2 (Architecture)** | `architecture.md` | **❌ MISMATCH** — defines `class GraphNet` |
| **Phase 3 (Dependency)** | `dependency.md` | **❌ PROPAGATION** — documents wrong interface |
| Phase 4 (Config) | `config.yaml` | ✓ Not implicated |
| Phase 5 (Analysis) | `analysis/` | ✓ Not implicated |
| **Phase 6 (Code)** | `models/gnn.py` | **❌ WRONG CLASS NAME** — implements `GNNModel` instead of `GraphNet` |

### Root Cause Chain

```
Phase 2 (Architecture)
  └── architecture.md specified: class GraphNet(models/gnn.py)
            │
            ▼  [Gap: Phase 2 gate may not have validated architecture-to-plan traceability]
Phase 6 (Code Generation)
  └── models/gnn.py implemented: GNNModel (NOT GraphNet)
            │
            ▼  [Phase 6 gate skipped: "verify imports resolve"]
Phase 7 (Execution)
  └── main.py imports: from models.gnn import GraphNet  ← FAILS
```

**The root cause is Phase 6 (code generation) producing `GNNModel` when Phase 2 (architecture) specified `GraphNet`.** The Phase 6 gate — which requires verifying imports resolve before moving on — was not enforced.

---

## Backtracking Decision

| Backtrack target | Rationale |
|-----------------|-----------|
| **Phase 6** (code) | Direct cause: `models/gnn.py` defines wrong class name |
| **Phase 2** (architecture) | Root cause upstream: architecture specified `GraphNet` but Phase 6 was allowed to deviate |

The failure is **not** simply a naming oversight in Phase 6. The architecture defined an explicit public interface (`GraphNet`). Phase 6 violated it without correcting the architecture. This means either:
1. Phase 6 did not read Phase 2 artefacts before generating code, OR
2. Phase 2's gate was not enforced — architecture was modified without propagating the change back

---

## Structured Debugging Plan

### Step 1: Verify Architecture-Plan Traceability (Phase 2 gate re-check)
- Read `architecture.md`
- Confirm `GraphNet` was the specified interface for `models/gnn.py`
- Confirm every experiment in `plan.md` traces to at least one module

### Step 2: Verify Architecture-Code Consistency (Phase 6 gate re-check)
- Read `models/gnn.py`
- Check: Did the generated code use `GNNModel` instead of `GraphNet`?
- Check: Did Phase 6 read `architecture.md` before implementing?
- Check: Are there any TODO or placeholder comments indicating unfinished work?

### Step 3: Inspect Dependency Model (Phase 3)
- Read `dependency.md`
- Was `models/gnn.GraphNet` listed as a dependency?
- Was the dependency list used during Phase 6 code generation?

### Step 4: Identify All Consumers of the Wrong Interface
Grep for all imports of `models.gnn` across the codebase:
```
from models.gnn import GraphNet   # main.py (failing)
from models.gnn import GNNModel  # (if any other file uses the wrong name)
```
List all files that import from `models.gnn` — all must be updated atomically.

### Step 5: Trace How `GNNModel` Was Introduced
- Check git log for `models/gnn.py` (if available)
- Did Phase 6 rename from `GraphNet` to `GNNModel` without updating architecture?
- Did Phase 5 analysis use a different name than Phase 2 architecture?

### Step 6: Fix and Re-validate
**Option A — Rename code to match architecture (if architecture is authoritative):**
1. Rename `class GNNModel` → `class GraphNet` in `models/gnn.py`
2. Verify all consumers import `GraphNet` correctly
3. Re-run Phase 7

**Option B — Update architecture to match code (if `GNNModel` is correct intent):**
1. Update `architecture.md` to specify `GNNModel` as the interface
2. Update `dependency.md` to reflect the correct class name
3. Update all imports across the codebase
4. Re-run Phase 7

**Decision criteria:**
- If `architecture.md` was generated from the paper and `GraphNet` is the paper-specified name → Option A
- If `GNNModel` is a sensible code-level refactor not in scope of the paper → Option B with documentation

### Step 7: Re-execute Phase 7
Run `main.py` and verify the import resolves.

---

## Why NOT Just Rename the Import

Simply changing `from models.gnn import GraphNet` to `from models.gnn import GNNModel` in `main.py` would patch the symptom but:
- Would leave the architecture-code mismatch unresolved
- Would break the traceable paper-to-implementation chain required by HiRAS
- Would leave `architecture.md` stating one thing and code doing another
- Would fail PaperBench's architecture-consistency checks

The Phase 2 → Phase 6 gate failure is the real issue. Either the architecture was wrong (Phase 2) or the code was wrong (Phase 6), and that must be diagnosed before patching imports.

---

## Phase to Revisit

**Primary: Phase 6 (code generation)** — the immediate source of the wrong class name.

**Secondary: Phase 2 (architecture)** — if Phase 6 did not consult Phase 2 artefacts, the manager checkpoint between Phase 2 and Phase 6 was bypassed.

The failure is **not** a Phase 3 (dependency) issue — `models/gnn` exists and is importable. The failure is a **Phase 2 ↔ Phase 6 inconsistency** at the interface level.