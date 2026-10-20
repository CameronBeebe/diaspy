"""Test that B and M format prompts identically where they should."""

import json


def test_b_and_m_format_example_blocks_identically():
    """B and M must format [i] {...} blocks identically for prosecutor prompts.
    
    This test reads the source code to verify the formatting string matches.
    """
    import experiments.theory_driven.arms.popperian as popperian_module
    import experiments.theory_driven.arms.millian as millian_module
    
    # Read the source files
    popperian_source = open(popperian_module.__file__).read()
    millian_source = open(millian_module.__file__).read()
    
    # B's format is at popperian.py around line 193:
    # f"[{i}] {json.dumps(ex, default=str)}"
    assert 'f"[{i}] {json.dumps(ex, default=str)}"' in popperian_source, \
        "B (Popperian) should format examples as [i] {...}"
    
    # M's format should be at millian.py around line 219:
    # f"[{i}] {json.dumps(row, default=str)}"
    assert 'f"[{i}] {json.dumps(row, default=str)}"' in millian_source, \
        "M (Millian) should format examples as [i] {...}"
    
    # Verify the pattern [i] is used (not [i} or [ i] or other variations)
    # The key part is the closing bracket after {i}
    assert '"[{i}] {' in popperian_source, "B must use [i] format with closing bracket"
    assert '"[{i}] {' in millian_source, "M must use [i] format with closing bracket"
    
    print("✓ B and M both format example blocks as [i] {...}")
