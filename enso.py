import argparse
import sys
import os
import subprocess
import json
import glob
from compiler import compile_source, analyze_source, set_verbose

BUILD_DIR = "__enso_build__"
CONFIG_FILE = "models.json"

def log(message):
    """Log informational messages to stderr."""
    print(f"\033[92m\033[1mINFO:\033[0m {message}", file=sys.stderr)

def cmd_update(args):
    latest_models = {
        "gpt-4o": { "type": "openai", "cost_in": 2.50, "cost_out": 10.00 },
        "gemini-flash-latest": { "type": "gemini", "cost_in": 3.50, "cost_out": 10.50 },
        "llama-3-local": { "type": "local", "cost_in": 0.0, "cost_out": 0.0 }
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(latest_models, f, indent=2)
    log(f"Updated {CONFIG_FILE} with {len(latest_models)} models.")

def build(filepath, verbose=False):
    set_verbose(verbose)
    
    if filepath is None or filepath == "-":
        # Read from stdin
        source_code = sys.stdin.read()
        filepath = "<stdin>"
    elif not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    else:
        with open(filepath, 'r') as f:
            source_code = f.read()
        
    if not os.path.exists(CONFIG_FILE):
        cmd_update(None)

    # Compile from source code
    try:
        python_code = compile_source(filepath, source_code if filepath == "<stdin>" else None)
    except Exception as e:
        print(f"Error: Compilation Failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    
    target_file = os.path.join(BUILD_DIR, "main.py")
    with open(target_file, 'w') as f:
        f.write(python_code)
        
    return target_file

def cmd_run(args):
    target_file = build(args.filename, verbose=args.verbose)
    # Check if main() exists in the compiled code
    with open(target_file, 'r') as f:
        code = f.read()
    
    if "def main(" in code:
        # Call main() if it exists
        trigger_code = "\n\nif __name__ == '__main__':\n    main()\n"
    else:
        # Otherwise just execute the script as-is (for backward compatibility with top-level statements)
        trigger_code = ""
    
    with open(target_file, 'a') as f:
        f.write(trigger_code)
    
    subprocess.call([sys.executable, target_file])

def cmd_test(args):
    target_file = build(args.filename, verbose=args.verbose)
    trigger_code = f"\n\nif __name__ == '__main__':\n    run_tests(include_ai={args.include_ai})\n"
    with open(target_file, 'a') as f:
        f.write(trigger_code)
    subprocess.call([sys.executable, target_file])

def cmd_analyse(args):
    """Analyze file: verify compilation, count AI calls, estimate cost without running them."""
    import json
    from datetime import datetime
    
    # Try to compile
    try:
        log("🔍 Analyzing file...")
        target_file = build(args.filename, verbose=args.verbose)
    except Exception as e:
        log(f"❌ Compilation failed: {e}")
        return
    
    # Read compiled code
    with open(target_file, 'r') as f:
        code = f.read()
    
    # Setup analysis environment
    analysis_results = {
        "filename": args.filename,
        "timestamp": datetime.now().isoformat(),
        "compilation": "success",
        "ai_calls": [],
        "execution_path": [],
        "models_used": {},
        "total_estimated_cost": 0.0,
        "realistic_tokens": args.realistic_tokens or False
    }
    
    # Parse out all AI function names from the compiled code
    import re
    # Look for: name="function_name"  (in the EnsoAgent constructor)
    ai_funcs = re.findall(r'name=["\'](\w+)["\']', code)
    
    # Find where to inject mocks - AFTER _ANALYSIS_RESULTS = None in the preamble
    # This is important because we don't want it overwritten by the preamble assignments
    inject_marker = "_ANALYSIS_RESULTS = None"
    inject_pos = code.find(inject_marker)
    
    if inject_pos >= 0:
        # Find the end of this line
        inject_line_end = code.find("\n", inject_pos) + 1
    else:
        # Fallback: find "_ANALYSIS_RESULTS = None" doesn't exist, use alternative
        inject_pos = code.find("ANALYSIS_MODE = False")
        if inject_pos >= 0:
            inject_line_end = code.find("\n", inject_pos) + 1
        else:
            inject_line_end = 0
    
    # Build mock injection code
    mocks_inject_code = "# Analysis mocks injected\nMOCKS.clear()\n"
    for func in ai_funcs:
        mocks_inject_code += f"class _MockResponse_{func}:\n"
        mocks_inject_code += f"    def __init__(self):\n"
        mocks_inject_code += f"        pass\n"
        mocks_inject_code += f"    def __getattr__(self, name):\n"
        mocks_inject_code += f"        return 'mocked'\n"
        mocks_inject_code += f"MOCKS['{func}'] = _MockResponse_{func}()\n"
    # Update _ANALYSIS_RESULTS - inject AFTER it's been set to None
    mocks_inject_code += f"_ANALYSIS_RESULTS = _imported_analysis_results\n"
    
    # Inject the mocks code after the _ANALYSIS_RESULTS = None line
    if inject_line_end > 0:
        code_with_mocks = code[:inject_line_end] + mocks_inject_code + code[inject_line_end:]
    else:
        code_with_mocks = code
    
    # Create analysis namespace with call tracking
    analysis_globals = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "_ANALYSIS_RESULTS": analysis_results,
        "_imported_analysis_results": analysis_results,  # Pass reference so injected code can use it
    }
    
    # Inject minimal mock for all print/log calls to suppress output during analysis
    def mock_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        analysis_results["execution_path"].append(f"[OUTPUT] {msg}")
    
    analysis_globals["print"] = mock_print
    
    # Execute with analysis tracking
    try:
        log("📊 Executing with automatic mocks...")
        exec(code_with_mocks, analysis_globals)
        
        # Call main() if it exists (to actually execute the AI functions)
        if "main" in analysis_globals:
            analysis_globals["main"]()
    except Exception as e:
        # Don't fail on execution errors during analysis - just report them
        analysis_results["execution_error"] = str(e)
        log(f"⚠️  Execution error during analysis (this is OK): {str(e)[:100]}")
    
    # Extract results
    ai_calls = analysis_results.get("ai_calls", [])
    execution_path = analysis_results.get("execution_path", [])
    
    # Generate report
    print("\n" + "="*60)
    print("           ENSO ANALYSIS REPORT")
    print("="*60 + "\n")
    
    # Compilation status
    print(f"📋 COMPILATION: ✓ Success\n")
    
    # AI Calls Summary
    if ai_calls:
        print(f"🤖 AI FUNCTION CALLS: {len(ai_calls)} total calls\n")
        
        # Build call breakdown by function and model
        call_breakdown = {}
        for call in ai_calls:
            func_name = call.get("function", "unknown")
            model = call.get("model", "unknown")
            key = func_name
            if key not in call_breakdown:
                call_breakdown[key] = {"count": 0, "model": model, "cost": 0.0}
            call_breakdown[key]["count"] += 1
            call_breakdown[key]["cost"] += call.get("cost", 0.0)
        
        # Print table
        print("CALL BREAKDOWN:")
        print("┌─────────────────────────┬───────┬──────────────────────┐")
        print("│ Function                │ Calls │ Model                │")
        print("├─────────────────────────┼───────┼──────────────────────┤")
        for func_name, info in sorted(call_breakdown.items()):
            func_display = func_name[:22].ljust(22)
            count_display = str(info["count"]).ljust(5)
            model_display = info["model"][:19].ljust(19)
            print(f"│ {func_display} │ {count_display} │ {model_display} │")
        print("└─────────────────────────┴───────┴──────────────────────┘\n")
        
        # Cost breakdown by model
        cost_by_model = {}
        for call in ai_calls:
            model = call.get("model", "unknown")
            cost = call.get("cost", 0.0)
            if model not in cost_by_model:
                cost_by_model[model] = {"cost": 0.0, "calls": 0}
            cost_by_model[model]["cost"] += cost
            cost_by_model[model]["calls"] += 1
        
        total_cost = sum(info["cost"] for info in cost_by_model.values())
        
        print("💰 ESTIMATED COST (if run with real API):")
        print("┌──────────────────┬───────┬─────────────┐")
        print("│ Model            │ Calls │ Estimated   │")
        print("├──────────────────┼───────┼─────────────┤")
        for model, info in sorted(cost_by_model.items()):
            model_display = model[:15].ljust(15)
            calls_display = str(info["calls"]).ljust(5)
            cost_display = f"${info['cost']:.6f}".ljust(10)
            print(f"│ {model_display} │ {calls_display} │ {cost_display} │")
        print("├──────────────────┼───────┼─────────────┤")
        print(f"│ TOTAL            │       │ ${total_cost:.6f}   │")
        print("└──────────────────┴───────┴─────────────┘\n")
        
        # Execution path (only if --show-path flag)
        if args.show_path and execution_path:
            print("🛤️  EXECUTION PATH:")
            for i, step in enumerate(execution_path[:20]):  # Show first 20 steps
                print(f"   {i+1}. {step}")
            if len(execution_path) > 20:
                print(f"   ... and {len(execution_path) - 20} more steps")
            print()
    else:
        print("🤖 AI FUNCTION CALLS: None detected\n")
    
    # Token estimation note
    if not args.realistic_tokens:
        print("⚠️  NOTE: Using minimal token estimates (256 in, 256 out)")
        print("   Use --realistic-tokens for more accurate cost prediction\n")
    else:
        print("✓ Using realistic token estimates for better accuracy\n")
    
    print("✨ No API calls were made during analysis\n")
    
    # Save report if requested
    if args.save_report:
        analysis_results["total_estimated_cost"] = total_cost if ai_calls else 0.0
        with open(args.save_report, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        log(f"📁 Report saved to {args.save_report}")

def cmd_init(args):
    log("Initializing project...")
    with open("main.enso", "w") as f:
        f.write('import "structs.enso";\n\nprint("Hello Enso");')
    with open("structs.enso", "w") as f:
        f.write('struct User { name: String }')
    log("Created main.enso and structs.enso")

def cmd_build(args):
    """Build Ensō files into a Python library package."""
    source = args.source
    name = args.name or "enso_lib"
    output_dir = args.output or "enso_out"
    
    # Find all .enso files
    if os.path.isfile(source):
        enso_files = [source]
    elif os.path.isdir(source):
        enso_files = sorted(glob.glob(os.path.join(source, "*.enso")))
    else:
        print(f"Error: '{source}' is neither a file nor directory", file=sys.stderr)
        sys.exit(1)
    
    if not enso_files:
        print(f"Error: No .enso files found in '{source}'", file=sys.stderr)
        sys.exit(1)
    
    log(f"Building library '{name}' with {len(enso_files)} file(s)...")
    
    # Compile all files and extract function info
    all_functions = []
    compiled_code = []
    
    for enso_file in enso_files:
        log(f"  Compiling {os.path.basename(enso_file)}...")
        try:
            # Analyze to get function signatures
            with open(enso_file, 'r') as f:
                source_code = f.read()
            functions = analyze_source(source_code)
            all_functions.extend(functions or [])
            
            # Compile the file
            code = compile_source(enso_file)
            compiled_code.append(code)
        except Exception as e:
            print(f"Error: Failed to compile {enso_file}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create output package directory
    package_dir = os.path.join(output_dir, name)
    os.makedirs(package_dir, exist_ok=True)
    
    # Write compiled code to _enso_runtime.py
    runtime_file = os.path.join(package_dir, "_enso_runtime.py")
    # Take the first compiled code (they all have preamble) and merge function definitions
    with open(runtime_file, 'w') as f:
        f.write(compiled_code[0])
    
    # Generate __init__.py with exported functions
    init_file = os.path.join(package_dir, "__init__.py")
    with open(init_file, 'w') as f:
        f.write(f'''"""Generated Ensō AI library: {name}
        
Provides AI functions for integration with Django, FastAPI, or other Python projects.
"""

from ._enso_runtime import (
    EnsoAgent,
    Ok,
    Err,
    AIError,
    ErrorKind,
    Result,
    Probabilistic,
)

# Import AI functions from runtime
''')
        
        # Add imports for each AI function
        for func in all_functions:
            if func and func.get('type') == 'function':
                f.write(f"from ._enso_runtime import {func['name']}\n")
        
        f.write('\n# Export all public APIs\n__all__ = [\n')
        f.write('    "EnsoAgent", "Ok", "Err", "AIError", "ErrorKind", "Result", "Probabilistic",\n')
        
        # Add each AI function to exports
        for func in all_functions:
            if func and func.get('type') == 'function':
                f.write(f"    '{func['name']}',\n")
        
        f.write(']\n')
    
    
    # Generate setup.py
    setup_file = os.path.join(output_dir, "setup.py")
    with open(setup_file, 'w') as f:
        f.write(f'''from setuptools import setup, find_packages

setup(
    name="{name}",
    version="0.1.0",
    description="Generated Ensō AI library",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
        "requests>=2.28",
    ],
)
''')
    
    # Generate requirements.txt
    req_file = os.path.join(output_dir, "requirements.txt")
    with open(req_file, 'w') as f:
        f.write("pydantic>=2.0\nrequests>=2.28\n")
    
    # Generate README.md
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, 'w') as f:
        f.write(f'''# {name}

Generated Ensō AI library.

## Installation

```bash
pip install -e .
```

## Usage

```python
from {name} import analyze, classify

# Use the AI functions
result = analyze("Your text here")
match result:
    case Ok(value):
        print(f"Result: {{value}}")
    case Err(error):
        print(f"Error: {{error.message}}")
```

## Available Functions

''')
        for func in all_functions:
            if func and func.get('type') == 'function':
                args = ", ".join([f"{a['name']}: {a['type']}" for a in func.get('args', [])])
                f.write(f"- `{func['name']}({args})` → {func.get('return', 'Result')}\n")
    
    log("Library built successfully!")
    log(f"📁 Output directory: {output_dir}/{name}")
    log(f"📋 Files generated:")
    log(f"   - _enso_runtime.py (compiled code)")
    log(f"   - __init__.py (exports)")
    log(f"   - setup.py (package config)")
    log(f"   - requirements.txt (dependencies)")
    log(f"   - README.md (documentation)")
    log(f"🚀 To use: pip install -e {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ensō CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (sent to stderr)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run file").add_argument("filename", nargs="?", default="-", help="File to run (default: read from stdin)")
    test = subparsers.add_parser("test", help="Run tests")
    test.add_argument("filename")
    test.add_argument("--include_ai", action="store_true")
    
    build = subparsers.add_parser("build", help="Build Python library from Ensō files")
    build.add_argument("source", help="Source .enso file or directory")
    build.add_argument("--name", help="Library name (default: enso_lib)")
    build.add_argument("--output", help="Output directory (default: enso_out)")
    
    analyse = subparsers.add_parser("analyse", help="Analyze file without running AI calls (cost estimation)")
    analyse.add_argument("filename")
    analyse.add_argument("--realistic-tokens", action="store_true", help="Use realistic token estimates for cost calculation")
    analyse.add_argument("--save-report", help="Save analysis report as JSON to file")
    analyse.add_argument("--show-path", action="store_true", help="Show execution path (can be verbose)")
    
    subparsers.add_parser("init", help="Init project")
    subparsers.add_parser("update", help="Update models")

    args = parser.parse_args()
    
    if args.command == "update": cmd_update(args)
    elif args.command == "run": cmd_run(args)
    elif args.command == "test": cmd_test(args)
    elif args.command == "build": cmd_build(args)
    elif args.command == "analyse": cmd_analyse(args)
    elif args.command == "init": cmd_init(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()