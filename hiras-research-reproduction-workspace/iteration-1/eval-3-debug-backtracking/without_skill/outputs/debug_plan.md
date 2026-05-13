# Debug Plan: TinyKAN-HAR Reproduction — `kan_layers` Module Not Found

## Error Summary
```
ModuleNotFoundError: No module named 'kan_layers'
```

## Root Cause Hypothesis

The generated code contains `from kan_layers import KANLayer` (or similar) but the actual module is defined in `models/kan.py` as `class KANLayer`. This is a **Phase 3 (dependency) / Phase 6 (code)** failure — the import statements reference a module name that doesn't match the actual file structure.

Common causes:
1. File named `models/kan.py` but import expects `kan_layers` module (plural vs singular)
2. File may be in wrong directory relative to Python path
3. `__init__.py` may be missing in the `models/` directory
4. Generated code assumed a `kan_layers/` subdirectory structure that wasn't created

---

## Debugging Steps

### Step 1: Locate the offending import
**Command:**
```bash
# Search for kan_layers imports across all generated files
rg "import kan_layers|from kan_layers" --type py .
rg "kan_layers" --type py .
```

**Expected finding:** Somewhere in the codebase (e.g., `models/kan.py`, `train.py`, or `tinykan_har/models/__init__.py`), there's an import statement expecting `kan_layers`.

---

### Step 2: Compare import expectation vs actual file structure
**Command:**
```bash
# List the models directory structure
ls -la models/        # or whatever directory contains kan.py
find . -name "kan*.py" -o -name "KAN*.py"
```

**Questions to answer:**
- Is `models/kan.py` the only KAN-related file?
- Is there supposed to be a `kan_layers.py` separate from `models/kan.py`?
- Is there a `kan_layers/` subdirectory that should exist?

---

### Step 3: Check `__init__.py` exports
If `models/kan.py` defines `KANLayer` but code imports `from kan_layers import KANLayer`, the `models/__init__.py` may need:
```python
from .kan import KANLayer
```

OR the import in the code needs to be fixed to:
```python
from models.kan import KANLayer
```

---

### Step 4: Verify Python path / package structure
```bash
# Check if the generated code is structured as a package
cat setup.py 2>/dev/null || cat pyproject.toml 2>/dev/null || echo "No setup files found"
python -c "import sys; print('\n'.join(sys.path))"
```

---

### Step 5: Apply the HiRAS Backtracking Rule

Per the HiRAS methodology (SKILL.md, backtracking rules):

| Failure type | Backtrack to |
|---|---|
| Import/module not found | **Phase 3 (dependency) or Phase 6 (code)** |

**Recommendation:** Backtrack to **Phase 3** to verify the dependency specification matched the actual code structure. The dependency analysis likely assumed `kan_layers` as a module name but the code generator created `models/kan.py` with `KANLayer`.

---

## Recommended Fixes

### Option A: Rename module (if `kan_layers.py` should exist)
If the intent was to have `kan_layers.py` (singular class, plural module), create `models/kan_layers.py` that re-exports:
```python
from .kan import KANLayer
```

### Option B: Fix import statements (preferred)
Update all imports to use the correct path:
```python
from models.kan import KANLayer
```
or ensure `models/__init__.py` properly exposes `KANLayer`.

### Option C: Check the code generator output
If using an automated code generator, the Phase 6 implementation may have deviated from the Phase 3 dependency specification. Re-run dependency analysis and verify the module naming convention.

---

## Verification

After fixing, verify with:
```bash
python -c "from models.kan import KANLayer; print('KANLayer imported successfully')"
python -c "import models.kan; print(dir(models.kan))"
```

---

## Metadata
- **Diagnosed by:** HiRAS evaluation agent
- **Phase of failure:** Phase 3 (dependency) / Phase 6 (code generation)
- **Next action:** Locate the mismatched import, determine correct fix (Option A, B, or C), apply, and re-run Phase 7 (execution)
