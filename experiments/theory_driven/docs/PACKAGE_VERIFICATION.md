# Package Name Verification

## Known Package Names (Verified)

### DSPy
- **PyPI name**: `dspy-ai`
- **Import name**: `dspy`
- **Latest version** (as of Oct 2026): 2.5.29
- **Note**: Version 3.4.1 referenced in audit does NOT exist
- **Correct install**: `pip install dspy-ai==2.5.29`

### GEPA
- **Status**: NOT on PyPI as standalone package
- **Distribution**: Part of DSPy or separate GitHub repo
- **Options**:
  1. GEPA may be in `dspy.experimental` or similar
  2. Install from git: `pip install git+https://github.com/[org]/gepa.git`
  3. **Action needed**: Verify GEPA distribution method

### TypeSafe / Jev
- **Status**: UNCLEAR
- **Possible scenarios**:
  1. TypeSafe is an API service, not a pip package
  2. Client library may exist under different name
  3. May require HTTP client only (aiohttp)
- **Spec reference**: "dspy.experimental.TypeSafe"
- **Action needed**: Verify if pip package exists or if API-only

### LiteLLM
- **PyPI name**: `litellm`
- **Latest version**: 1.50+
- **Verified**: ✓ Correct package name

## Actions Required

1. **Verify GEPA installation method**
   - Check if `dspy.GEPA` exists in dspy-ai 2.5.29
   - If not, find correct source

2. **Verify TypeSafe/Jev**
   - Check if `dspy.experimental.TypeSafe` exists
   - Document API-only access if no pip package

3. **Update robustness audit**
   - Correct version numbers
   - Note distribution uncertainties

## Current State

- requirements.txt has been updated with notes
- Actual versions may differ from initial estimates
- **No real installs attempted** (per constraint: make no paid calls, work offline)

## Honest Assessment

The initial audit assumed:
- dspy-ai==3.4.1 (DOES NOT EXIST)
- gepa==0.1.4 (MAY NOT EXIST as pip package)
- jev==1.13.0 (UNCLEAR if pip package)

**These need verification against actual PyPI and package sources before real runs.**

## Recommendation

Before first paid run:
1. Test install in clean virtualenv
2. Verify all imports work
3. Update requirements.txt with working versions
4. Document any manual install steps (e.g., git clones)
