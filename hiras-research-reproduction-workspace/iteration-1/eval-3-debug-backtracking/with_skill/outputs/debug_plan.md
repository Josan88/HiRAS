# HiRAS Debug Plan: `ModuleNotFoundError: No module named 'kan_layers'`

## 1. Error Classification

| Error | Root-cause phase(s) | Immediate action |
|---|---|---|
| `ModuleNotFoundError: No module named 'kan_layers'` | Phase 3 (dependency) **or** Phase 6 (code) | Inspect `dependency.md` and `models/kan.py` |

**Backtracking rule triggered:**
> Import/module not found → Phase 3 (dependency) or Phase 6 (code)

---

## 2. Diagnostic Steps

### Step 1 — Read `dependency.md` (Phase 3 output)

**What to look for:**
- Does `dependency.md` declare `kan_layers` as a top-level package/module?
- Does it list `models/kan` as the correct import path?
- Is the import relationship correctly specified (`models.kan` vs `kan_layers`)?

**Expected state:** For a file `models/kan.py` containing class `KANLayer`, code that imports it should use:
- `from models.kan import KANLayer` (relative import within the project), OR
- `from mypackage.models.kan import KANLayer` (absolute import with package prefix)

If `dependency.md` specifies `import kan_layers` as a top-level external dependency, the architecture planning agent (Phase 2) likely mis-framed the KAN library as an external package rather than an internal module.

### Step 2 — Read `models/kan.py` (Phase 6 output)

**What to look for:**
- Does the file define `KANLayer`? (confirmed by user)
- Does the file also define or export a module-level name `kan_layers`?
- Are there any `__init__.py` files in the `models/` directory that could cause `models.kan` to be hidden?

**If `models/__init__.py` re-exports something under a different name**, that explains the mismatch.

### Step 3 — Read `architecture.md` (Phase 2 output)

**What to look for:**
- Was `models/kan.py` placed under a package prefix (e.g., `src/` or the project package name)?
- What is listed as the project's top-level package name?
- Does the architecture specify `kan_layers` as an internal module name?

### Step 4 — Check `requirements.txt` or `config.yaml` (Phase 4)

**What to look for:**
- Is `kan_layers` listed as an external pip dependency?
- If the TinyKAN-HAR paper references a specific KAN library (e.g., `pykan` or `KAN`, or a custom `kan_layers` from a subpackage), confirm which one was intended.

---

## 3. Root Cause Hypothesis

Given the user description ("`models/kan.py` exists but it defines `KANLayer` instead of the expected module name"), the most likely causes in priority order:

### Hypothesis A (Phase 3 error): `dependency.md` specifies wrong import path
- `dependency.md` says `import kan_layers` but the project only has `models/kan.py` with `KANLayer`
- **Fix:** Update `dependency.md` to correctly specify `from models.kan import KANLayer` (or whatever the actual project package name is), then regenerate Phase 6 code that uses the correct import

### Hypothesis B (Phase 6 error): Code generator assumed wrong module name
- The code generator (Phase 6) produced `import kan_layers` in one or more files, but no such module exists
- The calling file(s) need to be updated to use `from models.kan import KANLayer`
- **Fix:** Re-run Phase 6 coding agent with corrected import instructions derived from `dependency.md`

### Hypothesis C (Phase 2 error): `architecture.md` placed module in wrong package
- Architecture specified `kan_layers` as a package name, but the coding agent placed the code in `models/kan.py`
- **Fix:** Update `architecture.md`, then re-derive `dependency.md`, then regenerate Phase 6 code

---

## 4. Recommended Backtracking Sequence

```
Execution (Phase 7) fails with ModuleNotFoundError
         ↓
Inspect dependency.md → Is 'kan_layers' listed as external pip dep?
    YES → Remove it from external deps; add correct internal import path
    NO  → Continue to Step 2
         ↓
Inspect models/kan.py → Does it export 'kan_layers'?
    YES → The calling files import the wrong name; fix them
    NO  → Continue to Step 3
         ↓
Inspect architecture.md → Was 'kan_layers' specified as a package?
    YES → Rename module to match OR update architecture
    NO  → Continue to Step 4
         ↓
Inspect requirements.txt/config.yaml → Which KAN library is referenced?
    → Confirm paper's KAN library (pykan? custom kan_layers?)
    → Install correct package or correct internal import path
```

---

## 5. Specific Fix Actions

### Fix A — Correct the dependency model (Phase 3 revision)
1. Read `dependency.md`
2. Remove any entry treating `kan_layers` as an external pip package
3. Add correct inter-file import: e.g., `models.kan → KANLayer`
4. Re-run dependency planning agent to propagate corrected deps to all components

### Fix B — Correct the import statements in calling code (Phase 6 revision)
1. Find all files that contain `import kan_layers` or `from kan_layers import`
2. Replace with the correct import path (e.g., `from models.kan import KANLayer`)
3. If `models/` is a package, verify/create `models/__init__.py` with appropriate exports

### Fix C — Verify/create `models/__init__.py` (Phase 6, often skipped)
1. If `models/` is a package (has `__init__.py`), check whether `__init__.py` re-exports `KANLayer`
2. If missing, add: `from .kan import KANLayer` to make `from models import KANLayer` work
3. This bridges the gap between `models.kan.KANLayer` and `models.KANLayer` import styles

---

## 6. Phase Responsibility Summary

| Phase | Artefact | Failure here? | Action |
|---|---|---|---|
| Phase 2 | `architecture.md` | Possibly — wrong package/module naming | Revise architecture |
| Phase 3 | `dependency.md` | **Likely** — `kan_layers` wrongly listed as external or internal module | Revise dependencies |
| Phase 6 | `models/kan.py` + callers | Possibly — wrong import statement in calling code | Re-code callers |
| Phase 7 | Execution | Observed failure | Re-execute after fixes |

**Primary suspect:** Phase 3 — the `dependency.md` was either missing the inter-file import relationship or mis-identified `kan_layers` as a separate top-level module rather than a class within `models.kan`.

---

## 7. Execution Checklist After Fix

- [ ] `dependency.md` no longer references `kan_layers` as an external module
- [ ] All `import kan_layers` / `from kan_layers import` replaced with correct path
- [ ] `models/__init__.py` exports `KANLayer` if `from models import KANLayer` style is used
- [ ] `python -c "from models.kan import KANLayer"` succeeds in the project environment
- [ ] Re-run experiment(s) from the plan to confirm Phase 7 passes