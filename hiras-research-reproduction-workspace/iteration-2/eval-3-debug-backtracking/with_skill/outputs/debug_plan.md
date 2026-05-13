# HiRAS Debug Plan: `ModuleNotFoundError: No module named 'kan_layers'`

## 1. Error Classification

| Error | Root-cause phase(s) | Immediate action |
|---|---|---|
| `ModuleNotFoundError: No module named 'kan_layers'` | **Phase 3 (dependency)** primary; **Phase 6 (code)** secondary | Inspect `dependency.md` and all files containing `import kan_layers` |

**Backtracking rule triggered:**
> Import/module not found → Phase 3 (dependency) or Phase 6 (code)

---

## 2. Diagnostic Steps

### Step 1 — Read `dependency.md` (Phase 3 output)

**What to look for:**
- Does `dependency.md` declare `kan_layers` as a top-level external package?
- Does it correctly specify `models.kan.KANLayer` as an inter-file dependency?
- Is the import relationship between the caller file and `models/kan.py` documented?

**Expected state:** For a file `models/kan.py` containing class `KANLayer`, any caller file should use:
- `from models.kan import KANLayer` (relative import within the project), OR
- `from tinykan_har.models.kan import KANLayer` (absolute import with package prefix)

If `dependency.md` specifies `import kan_layers` as a pip-installable or top-level external dependency, the dependency planning agent mis-framed the KAN component as an external library rather than an internal module.

### Step 2 — Find all files that import `kan_layers` (Phase 6 output)

**What to look for:**
- Grep the repository for `import kan_layers` or `from kan_layers import`
- Identify the caller files that use the wrong module name
- Determine which caller file first introduced `import kan_layers`

### Step 3 — Read `models/kan.py` (Phase 6 output)

**What to look for:**
- Does the file define `KANLayer`? (confirmed by user)
- Does the file also define or export a module-level name `kan_layers`?
- Are there any `models/__init__.py` re-exports under a different name?

### Step 4 — Read `architecture.md` (Phase 2 output)

**What to look for:**
- Was `models/kan.py` placed under the project package `tinykan_har`?
- Does the architecture specify `kan_layers` as an internal module name? (It should NOT)
- The architecture at line 19 names the file `kan_layer.py` (singular), but the user reports `models/kan.py` (singular) — verify if this naming mismatch is part of the problem.

---

## 3. Root Cause Hypothesis

Based on the HiRAS backtracking table and the evidence from prior phases:

### Hypothesis A (Phase 3 error — PRIMARY): `dependency.md` specifies wrong import path
- `dependency.md` says `import kan_layers` but the project only has `models/kan.py` with `KANLayer`
- The dependency model either listed `kan_layers` as an external pip package or failed to document the inter-file relationship
- **Fix:** Revise `dependency.md` to correctly specify `from tinykan_har.models.kan import KANLayer` (or whatever the project package name is), then regenerate Phase 6 code using corrected imports

### Hypothesis B (Phase 6 error — SECONDARY): Code generator assumed wrong module name
- Phase 6 code generator produced `import kan_layers` in one or more files, but no such module exists
- The calling file(s) need to be updated to use `from models.kan import KANLayer`
- **Fix:** Re-run Phase 6 coding agent with corrected import instructions from corrected `dependency.md`

### Hypothesis C (Phase 2 error): Architecture named the file `kan_layer.py` but code produced `kan.py`
- The architecture says `models/kan_layer.py` but the actual file is `models/kan.py`
- This mismatch suggests Phase 6 code generation deviated from the architecture
- **Fix:** Reconcile architecture.md with actual file structure, then propagate corrections through Phases 3 and 6

---

## 4. Recommended Backtracking Sequence

```
Execution (Phase 7) fails with ModuleNotFoundError: 'kan_layers'
         ↓
Inspect dependency.md → Is 'kan_layers' listed as external pip dep or wrong inter-file path?
     YES → Remove; specify correct path: from tinykan_har.models.kan import KANLayer
     NO  → Continue to Step 2
         ↓
Grep for 'import kan_layers' / 'from kan_layers import' → find all violating caller files
     Found → Replace with correct import from dependency.md
     Not found → Suspicious; check if kan_layers appears in any generated file
         ↓
Inspect architecture.md → Was 'kan_layers' specified as a package/module name?
     YES → Remove from architecture; clarify 'models.kan' and 'KANLayer' are the correct references
     NO  → Continue to Step 4
         ↓
Verify/create models/__init__.py → ensure KANLayer is re-exported
     → Add: from .kan import KANLayer (if missing)
```

---

## 5. Phase Responsibility Summary

| Phase | Artefact | Failure here? | Action |
|---|---|---|---|
| Phase 0 | Intake | No — paper read correctly | None |
| Phase 1 | `plan.md` | No — all 6 experiments covered | None |
| Phase 2 | `architecture.md` | **Possibly** — file naming mismatch (`kan_layer.py` vs `kan.py`) | Revise architecture |
| Phase 3 | `dependency.md` | **Likely** — `kan_layers` wrongly treated as external module | Revise dependencies |
| Phase 4 | `config.yaml` | No — not relevant to import error | None |
| Phase 5 | `analysis/` | No — analysis done before code gen | None |
| Phase 6 | Code files | **Possibly** — generated `import kan_layers` in caller files | Fix import statements |
| Phase 7 | Execution | Observed failure — cannot proceed until Phases 2/3/6 are corrected | Re-execute after fixes |

**Primary suspect:** Phase 3 — the `dependency.md` was either missing the inter-file import relationship or mis-identified `kan_layers` as a separate top-level module rather than a class within `models.kan`.

**Secondary suspect:** Phase 6 — code generator may have used `import kan_layers` instead of `from tinykan_har.models.kan import KANLayer` in one or more caller files.

---

## 6. Specific Fix Actions

### Fix A — Correct the dependency model (Phase 3 revision)
1. Read `dependency.md`
2. Remove any entry treating `kan_layers` as an external pip package
3. Add correct inter-file import: `tinykan_har.models.kan → KANLayer` (exported to callers)
4. Re-run dependency planning agent to propagate corrected deps to all components

### Fix B — Correct the import statements in calling code (Phase 6 revision)
1. Grep for all `import kan_layers` and `from kan_layers import` in generated code
2. Replace each occurrence with `from tinykan_har.models.kan import KANLayer` (or correct project package path)
3. Verify `models/__init__.py` re-exports `KANLayer` if `from models import KANLayer` style is used

### Fix C — Reconcile architecture and file naming (Phase 2 + Phase 6)
1. Compare `architecture.md` module list with actual generated files
2. If `architecture.md` says `kan_layer.py` but actual file is `kan.py`, decide which is canonical
3. Update architecture.md to match the chosen convention
4. Ensure all references (dependency.md, code files) use the agreed name consistently

### Fix D — Verify/create `models/__init__.py`
1. Check whether `models/__init__.py` exists
2. If it exists, verify it re-exports `KANLayer` appropriately
3. If missing, add: `from .kan import KANLayer` to enable `from models import KANLayer`
4. This bridges the gap between `models.kan.KANLayer` and `models.KANLayer` import styles

---

## 7. Execution Checklist After Fix

- [ ] `dependency.md` no longer references `kan_layers` as an external module
- [ ] All `import kan_layers` / `from kan_layers import` replaced with correct path in all generated files
- [ ] `architecture.md` file names match actual generated file names (reconcile `kan_layer.py` vs `kan.py`)
- [ ] `models/__init__.py` exports `KANLayer` if `from models import KANLayer` style is used
- [ ] `python -c "from tinykan_har.models.kan import KANLayer"` succeeds in the project environment (adjust import path to match actual project package name)
- [ ] Re-run Phase 7 execution to confirm the import error is resolved before proceeding with experiments

---

## 8. Key Ambiguity to Resolve

**Which KAN library was intended by the paper?**
- The paper's methodology may reference an external KAN library (e.g., `pykan`, or a specific `kan_layers` from a cited repository)
- If the paper explicitly uses an external `kan_layers` package, then `pip install` it as a dependency
- If the paper describes KAN as a custom module to be implemented, then `kan_layers` should not exist as an external import — only as an internal class `KANLayer` in `models/kan.py`
- **Resolution:** Check the paper's Section 3 (Methodology) for any citation of a KAN library. If none cited, treat KAN as a custom implementation and ensure all internal imports reference `models.kan`

---

## 9. Next Steps if Backtracking Revisits Earlier Phases

| If backtracking to | Action |
|---|---|
| Phase 2 | Revise `architecture.md` to fix file naming (`kan_layer.py` vs `kan.py`), re-export interface contracts |
| Phase 3 | Regenerate `dependency.md` with correct inter-file import map; no external `kan_layers` unless paper cites it |
| Phase 6 | Regenerate all caller files using the corrected import paths from updated `dependency.md` |
| Phase 0 | Only if the paper's KAN description suggests a different implementation approach than currently planned |