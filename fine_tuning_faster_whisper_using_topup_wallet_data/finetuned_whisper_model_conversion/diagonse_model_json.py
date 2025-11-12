"""
Diagnose all JSON files in the model directory to find corrupted ones
"""

import os
import json

MODEL_DIR = r"faster-whisper-finetuned-topupwallet"

print("=" * 70)
print("CHECKING ALL JSON FILES IN MODEL DIRECTORY")
print("=" * 70)
print(f"Directory: {MODEL_DIR}\n")

if not os.path.exists(MODEL_DIR):
    print(f"ERROR: Directory does not exist: {MODEL_DIR}")
    exit(1)

# List all files
print("Files in directory:")
for file in os.listdir(MODEL_DIR):
    file_path = os.path.join(MODEL_DIR, file)
    size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
    print(f"  - {file:40s} ({size:>12,} bytes)")

print("\n" + "=" * 70)
print("VALIDATING JSON FILES")
print("=" * 70)

json_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.json')]

if not json_files:
    print("No JSON files found!")
else:
    for json_file in json_files:
        file_path = os.path.join(MODEL_DIR, json_file)
        print(f"\n[{json_file}]")
        print("-" * 70)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Show first few lines
            lines = content.split('\n')
            print(f"First 5 lines:")
            for i, line in enumerate(lines[:5], 1):
                print(f"  {i}: {line[:100]}")
            
            # Try to parse
            parsed = json.loads(content)
            print(f"✓ VALID JSON")
            print(f"  Keys: {list(parsed.keys())[:10]}")  # Show first 10 keys
            
        except json.JSONDecodeError as e:
            print(f"✗ INVALID JSON!")
            print(f"  Error: {e}")
            print(f"  Line {e.lineno}, Column {e.colno}")
            
            # Show the problematic line
            if e.lineno <= len(lines):
                problem_line = lines[e.lineno - 1]
                print(f"  Problematic line: {problem_line}")
                print(f"  Position marker: {' ' * (e.colno - 1)}^")
            
            # Show context around error
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            print(f"\n  Context (lines {start+1}-{end}):")
            for i in range(start, end):
                marker = " >>> " if i == e.lineno - 1 else "     "
                print(f"{marker}{i+1}: {lines[i][:100]}")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)