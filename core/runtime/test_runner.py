# ==========================================
# TEST RUNNER
# ==========================================

def run_tests(include_ai=False):
    """
    Run all test functions in the compiled module.
    
    Args:
        include_ai: If True, run AI tests (which make real API calls).
                   If False, skip AI tests (only run mocked tests).
    """
    print(f"\n🧪 Running Tests (Include AI: {include_ai})...", file=sys.stderr)
    g = globals()
    tests = [name for name in g if name.startswith("test_")]
    passed = 0
    skipped = 0
    for t in tests:
        is_ai_test = "_AI_" in t
        print(t, file=sys.stderr)
        if is_ai_test and not include_ai:
            skipped += 1
            continue
        MOCKS.clear()
        try:
            g[t]()
            print(f"✅ PASS: {t}", file=sys.stderr)
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {t} ({e})", file=sys.stderr)
    print(f"\nSummary: {passed} Passed, {skipped} Skipped.", file=sys.stderr)
