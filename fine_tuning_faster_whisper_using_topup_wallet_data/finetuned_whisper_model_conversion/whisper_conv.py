"""
Simple CPU-only converter without Unicode characters
"""

import os
import sys

# DISABLE CUDA BEFORE IMPORTING ANYTHING
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Now import
import torch
from ctranslate2.converters import TransformersConverter

# Paths
MODEL_PATH = r"faster-whisper-finetuned-topupwallet"
OUTPUT_PATH = r"faster-whisper-malaysian"

print("=" * 70)
print("CPU-ONLY CONVERSION")
print("=" * 70)

# Check CUDA is disabled
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("ERROR: CUDA is still enabled!")
    print("Try running: set CUDA_VISIBLE_DEVICES=-1")
    sys.exit(1)

print("CUDA successfully disabled")
print("\nStarting conversion (this may take 5-10 minutes)...")
print("=" * 70)

try:
    # Create converter
    converter = TransformersConverter(MODEL_PATH)
    
    # Convert
    converter.convert(
        output_dir=OUTPUT_PATH,
        quantization="float16",
        force=True
    )
    
    print("\n" + "=" * 70)
    print("SUCCESS! CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {OUTPUT_PATH}")
    
    # List files
    if os.path.exists(OUTPUT_PATH):
        print("\nFiles created:")
        for file in os.listdir(OUTPUT_PATH):
            file_path = os.path.join(OUTPUT_PATH, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file:30s} ({size:>8.2f} MB)")
    
    print("\n" + "=" * 70)
    print("TEST YOUR MODEL:")
    print("=" * 70)
    print("""
from faster_whisper import WhisperModel

model = WhisperModel(
    r"G:\\Vanilla_Tech\\speech2text\\feat-maintained-code\\jwt_implementation_sst\\speech2text--ai\\models\\faster-whisper-malaysian",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe("audio.wav", language="ms")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
""")

except Exception as e:
    print("\n" + "=" * 70)
    print("CONVERSION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print("\nRecommendation: Use Transformers directly instead of faster-whisper")
    print("See the 'Use Model Directly with Transformers' artifact")
    sys.exit(1)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

# ct2-transformers-converter --model whisper-finetuned-topupwallet --output_dir faster-whisper-malaysian-final --copy_files "tokenizer.json" --quantization float16
