import argparse
import sys
import os
import subprocess
import json
from compiler import compile_source

BUILD_DIR = "__enso_build__"
CONFIG_FILE = "models.json"

def cmd_update(args):
    latest_models = {
        "gpt-4o": { "type": "openai", "cost_in": 2.50, "cost_out": 10.00 },
        "gemini-flash-latest": { "type": "gemini", "cost_in": 3.50, "cost_out": 10.50 },
        "llama-3-local": { "type": "local", "cost_in": 0.0, "cost_out": 0.0 }
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(latest_models, f, indent=2)
    print(f"✅ Updated {CONFIG_FILE} with {len(latest_models)} models.")

def build(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    if not os.path.exists(CONFIG_FILE):
        cmd_update(None)

    # Compile from file path directly (handles imports)
    try:
        python_code = compile_source(filepath)
    except Exception as e:
        print(f"❌ Compilation Failed:\n{e}")
        sys.exit(1)

    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
    
    target_file = os.path.join(BUILD_DIR, "main.py")
    with open(target_file, 'w') as f:
        f.write(python_code)
        
    return target_file

def cmd_run(args):
    target_file = build(args.filename)
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
    target_file = build(args.filename)
    trigger_code = f"\n\nif __name__ == '__main__':\n    run_tests(include_ai={args.include_ai})\n"
    with open(target_file, 'a') as f:
        f.write(trigger_code)
    subprocess.call([sys.executable, target_file])

def cmd_init(args):
    print("Initializing project...")
    with open("main.enso", "w") as f:
        f.write('import "structs.enso";\n\nprint("Hello Enso");')
    with open("structs.enso", "w") as f:
        f.write('struct User { name: String }')
    print("Created main.enso and structs.enso")

def main():
    parser = argparse.ArgumentParser(description="Ensō CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run file").add_argument("filename")
    test = subparsers.add_parser("test", help="Run tests")
    test.add_argument("filename")
    test.add_argument("--include_ai", action="store_true")
    subparsers.add_parser("init", help="Init project")
    subparsers.add_parser("update", help="Update models")

    args = parser.parse_args()
    
    if args.command == "update": cmd_update(args)
    elif args.command == "run": cmd_run(args)
    elif args.command == "test": cmd_test(args)
    elif args.command == "init": cmd_init(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()