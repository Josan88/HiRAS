# Debug Plan: PaperBench — `ImportError: cannot import name 'GraphNet' from 'models.gnn'`

## Error Summary

```
ImportError: cannot import name 'GraphNet' from 'models.gnn'
```

Executed via: `python main.py`

---

## Root Cause Diagnosis

The error is a **class name mismatch** across three artefacts:

| Artefact | Declares | Actual |
|---|---|---|
| `architecture.md` | `class GraphNet` in `models/gnn.py` | — |
| `models/gnn.py` (code) | — | defines `GNNModel` (not `GraphNet`) |
| `main.py` (entry point) | `from models.gnn import GraphNet` | expects `GraphNet` |

**The coding agent implemented `GNNModel` but the architecture specified `GraphNet`**, and the entry point was generated to match the architecture spec (not the actual implementation). This is a classic Phase 6 (code generation) inconsistency — the generated code's class name diverged from the architecture without propagating that change back to dependent code.

**This is NOT merely a renaming problem.** The correct fix is **not** to blindly rename `GNNModel → GraphNet`. The root question is: which artefact reflects the paper's intended interface? If the paper's methodology defines the interface as `GraphNet`, the code must provide `GraphNet`. If the code chose `GNNModel` for a legitimate reason (e.g., avoiding namespace collision), then the architecture is wrong. The debug plan below addresses both possibilities.

---

## Phase Backtracking

| Failure type | Backtrack to |
|---|---|
| Import/module not found | Phase 3 (dependency) or Phase 6 (code) |

**Backtrack to Phase 6 (code) first**, then reassess Phase 3 (dependency).

---

## Debugging Steps

### Step 1: Confirm the exact class name in `models/gnn.py`

Read `models/gnn.py` (or equivalent path) and identify:
- The exact class name defined (is it `GNNModel` or something else?)
- Whether it inherits from `torch.nn.Module`
- Its `__init__` signature and public methods

```python
# Expected pattern:
class GNNModel(torch.nn.Module):
    def __init__(self, ...):
        ...
```

### Step 2: Check what `main.py` actually imports

```python
# In main.py, find the import line:
from models.gnn import GraphNet  # ← this fails

# Also check: does main.py import anything else from models.gnn?
```

### Step 3: Check `models/__init__.py` for re-exports

```python
# models/__init__.py may re-export a different name
from .gnn import GNNModel  # or GraphNet
```

If `models/__init__.py` exports `GNNModel` but `main.py` imports `GraphNet`, the mismatch is at the package level — not just within `gnn.py`.

### Step 4: Verify architecture.md class name specification

Find the architecture's definition of `GraphNet` in `architecture.md`. The architecture should specify:
- Class name
- Module location (`models/gnn.py`)
- Public interface (methods, `__init__` signature)
- Role in the pipeline

If the architecture says `class GraphNet(models.gnn)`, then the code's `GNNModel` is the deviation. If the architecture says `class GNNModel`, then the import in `main.py` is the deviation.

### Step 5: Trace import dependency

Determine which file was generated first and which agent produced it:
1. `architecture.md` → specifies `GraphNet` interface
2. `models/gnn.py` → coding agent implements as `GNNModel` (unreported divergence)
3. `main.py` → generated against architecture spec, imports `GraphNet`
4. Execution → fails

The divergence occurred at the coding step when the agent renamed the class without updating the architecture or notifying the manager.

---

## Recommended Fixes

### Fix A — Align code to architecture (if architecture reflects paper intent)

Rename `GNNModel` → `GraphNet` in `models/gnn.py`:

```python
# models/gnn.py
class GraphNet(torch.nn.Module):  # was: GNNModel
    def __init__(self, ...):
        ...
```

Then also update `models/__init__.py` if it re-exports the class:
```python
from .gnn import GraphNet  # was: from .gnn import GNNModel
```

**Then re-run:** `python main.py`

This fix applies when the architecture spec (`GraphNet`) is the authoritative interface and the code deviated.

### Fix B — Align architecture to code (if code's choice is technically justified)

If the coding agent had a legitimate reason to rename (e.g., `GNNModel` avoids confusion with a library class named `GraphNet`), update `architecture.md` and regenerate the import statements:

In `architecture.md`, change the class specification from `GraphNet` to `GNNModel`.

In `main.py`, update the import:
```python
from models.gnn import GNNModel  # was: from models.gnn import GraphNet
```

**Then re-run:** `python main.py`

This fix applies when the code's naming is correct and the architecture spec was an error.

### Fix C — Add an alias in `models/gnn.py` (lowest-risk fast fix)

If both `GraphNet` and `GNNModel` need to coexist (e.g., both names are referenced in different parts of the codebase), add an alias without changing existing code:

```python
# models/gnn.py
class GNNModel(torch.nn.Module):
    ...

GraphNet = GNNModel  # Alias so both names resolve
```

This preserves existing code while making `GraphNet` available for import. However, this is a temporary patch — the underlying inconsistency should still be resolved properly.

---

## Verification Commands

```bash
# After applying fix:
python -c "from models.gnn import GraphNet; print('GraphNet OK')"
python -c "from models.gnn import GNNModel; print('GNNModel OK')"
python main.py  # should not fail on import
```

---

## Broader Context: Why Execution Scored 20%

The 20% execution score and 10% result match indicate this is not an isolated import error — it is the **first visible symptom** of a deeper Phase 6 (code generation) inconsistency. When `main.py` fails to import, the entire execution pipeline halts, preventing any experiment from running. This cascades:
- No training runs → no results → result match = 10% (trivial pass from partial outputs)
- No model instantiation → code development also stunted at 80%

The 80% code development score suggests some modules compiled and passed their unit tests in isolation, but the end-to-end entry point (`main.py`) was never tested until the rubric evaluation ran it.

---

## Metadata

- **Error class:** `ImportError` (import-time failure, Phase 6)
- **Phase of failure:** Phase 6 (code generation) — class name mismatch between architecture spec and implementation
- **HiRAS backtracking rule:** Import/module not found → backtrack to Phase 3 or Phase 6
- **Fix scope:** 1 class rename in 1–2 files (`models/gnn.py`, `models/__init__.py`) + import in `main.py`
- **Priority:** Critical — blocks all execution