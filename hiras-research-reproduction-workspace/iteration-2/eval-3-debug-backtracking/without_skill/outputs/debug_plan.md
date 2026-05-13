# Debug Plan: TinyKAN-HAR — `ModuleNotFoundError: No module named 'kan_layers'`

## Error Summary

```
ModuleNotFoundError: No module named 'kan_layers'
```

Generated code contains `from kan_layers import KANLayer` (or similar) but the
actual class `KANLayer` lives in `models/kan.py`. This is a naming mismatch between
the expected module name (`kan_layers`) and the actual file/module name.

---

## Root Cause Diagnosis

| Layer | Location | Issue |
|---|---|---|
| **Phase 6 (code generation)** | Somewhere in generated code | Import statement uses `kan_layers` module name that was never created |
| **Phase 3 (dependency)** | Dependency specification | Likely listed `kan_layers` as a module but actual code created `models/kan.py` instead |
| **Phase 2 (architecture)** | Architecture design | May have assumed a flat `kan_layers.py` file vs. `models/kan.py` structure |

**Backtrack to:** Phase 3 (dependency) and Phase 6 (code) — the import expectation
was never matched by the actual file structure.

---

## Debugging Steps

### Step 1: Locate every `kan_layers` import

```bash
# Search all Python files for kan_layers imports
rg "import kan_layers|from kan_layers" --type py D:\HiRAS
rg "\bkan_layers\b" --type py D:\HiRAS
```

Also search for any `KANLayer` import that doesn't use a `models.` prefix:
```bash
rg "KANLayer" --type py D:\HiRAS
```

### Step 2: Confirm the actual file structure

```bash
# Find all KAN-related Python files
Get-ChildItem -Path D:\HiRAS -Recurse -Include "*.py" | Where-Object { $_.Name -match "kan|KAN" }
```

Expected findings:
- `models/kan.py` with class `KANLayer` ← confirmed by user
- Possibly a `models/__init__.py` that may or may not re-export `KANLayer`
- NO `kan_layers.py` or `kan_layers/` directory

### Step 3: Check `models/__init__.py`

```python
# models/__init__.py should either:
# Option A — re-export KANLayer:
from .kan import KANLayer

# Option B — import the module itself:
from . import kan
```

If `models/__init__.py` is missing or empty, that explains why `from models.kan
import KANLayer` also fails.

### Step 4: Trace the import chain in the generated code

Find the file that first calls `from kan_layers import ...` and determine:
1. Is it trying to import as a top-level package (`kan_layers`) or a relative
   module (`from .kan_layers` or `from . import kan_layers`)?
2. Does the generated `models/` directory have an `__init__.py` at all?
3. Is there a `kan_layers.py` file that was supposed to be created but wasn't?

### Step 5: Verify Python path / execution context

```bash
# What directory is being run from?
pwd  # or Get-Location

# Is D:\HiRAS in sys.path?
python -c "import sys; print('D:\HiRAS' in sys.path)"

# Try importing the class directly via the confirmed path
python -c "from models.kan import KANLayer; print('OK')"
```

---

## Recommended Fixes

### Option A: Create `kan_layers.py` shim (if external code expects `kan_layers`)

Create `models/kan_layers.py`:
```python
from .kan import KANLayer
__all__ = ["KANLayer"]
```

Then ensure `models/__init__.py` also exposes it:
```python
from .kan_layers import KANLayer
```

### Option B: Fix all import statements to use correct path

In every generated file that has:
```python
from kan_layers import KANLayer   # wrong
```

Replace with:
```python
from models.kan import KANLayer   # correct
```

Or, if `models/__init__.py` properly re-exports:
```python
from models import KANLayer
```

### Option C: Add missing `__init__.py` to `models/` directory

Create `models/__init__.py`:
```python
from .kan import KANLayer
```

This allows `from models import KANLayer` everywhere.

---

## Verification Commands

```bash
# After applying fix:
python -c "from models.kan import KANLayer; print('models.kan.OK')"
python -c "import models; print(hasattr(models, 'KANLayer'))"
python -c "from models import KANLayer; print('OK')"
```

---

## Metadata

- **Error class:** `ModuleNotFoundError` (import-time failure)
- **Phase of failure:** Phase 3 (dependency mismatch) → Phase 6 (code generation)
- **HiRAS backtracking rule:** Import/module not found → backtrack to Phase 3 or Phase 6
- **Fix scope:** Likely 1-3 lines across `models/__init__.py` and/or import statements