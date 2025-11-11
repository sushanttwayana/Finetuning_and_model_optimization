"""
Diagnostic script to identify issues with converted Whisper models
Run this to diagnose why your fine-tuned model isn't loading
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_file_sizes(model_dir):
    """Check and report file sizes"""
    logger.info("\n" + "="*70)
    logger.info("📊 FILE SIZE ANALYSIS")
    logger.info("="*70)
    
    total_size = 0
    files_found = []
    
    for filename in sorted(os.listdir(model_dir)):
        filepath = os.path.join(model_dir, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            total_size += size
            files_found.append((filename, size_mb))
            
            # Flag suspicious sizes
            warning = ""
            if filename == "model.bin" and size_mb < 100:
                warning = " ⚠️  TOO SMALL - Should be ~1-3GB for large-v3"
            elif filename == "model.bin" and size_mb > 5000:
                warning = " ⚠️  TOO LARGE - Unexpected size"
            elif size < 100:
                warning = " ⚠️  Very small file"
            
            logger.info(f"  {filename:30s} {size_mb:>10.2f} MB{warning}")
    
    total_mb = total_size / (1024 * 1024)
    total_gb = total_mb / 1024
    
    logger.info("-" * 70)
    logger.info(f"Total: {total_mb:.2f} MB ({total_gb:.2f} GB)")
    logger.info("="*70)
    
    return files_found


def check_required_files(model_dir):
    """Check if all required files exist"""
    logger.info("\n" + "="*70)
    logger.info("📋 REQUIRED FILES CHECK")
    logger.info("="*70)
    
    # Critical files
    critical_files = {
        "model.bin": "CTranslate2 model weights",
        "config.json": "Model configuration"
    }
    
    # Tokenizer files (at least one needed)
    tokenizer_files = {
        "tokenizer.json": "HuggingFace tokenizer",
        "vocab.json": "Vocabulary (GPT-2 style)",
        "vocabulary.json": "Vocabulary (alternative)",
        "merges.txt": "BPE merges"
    }
    
    # Additional helpful files
    optional_files = {
        "preprocessor_config.json": "Audio preprocessing config",
        "special_tokens_map.json": "Special tokens mapping",
        "tokenizer_config.json": "Tokenizer configuration",
        "normalizer.json": "Text normalizer",
        "added_tokens.json": "Additional tokens"
    }
    
    all_ok = True
    
    # Check critical files
    logger.info("\n🔴 CRITICAL FILES (must exist):")
    for filename, description in critical_files.items():
        exists = os.path.exists(os.path.join(model_dir, filename))
        status = "✅" if exists else "❌ MISSING"
        logger.info(f"  {status} {filename:30s} - {description}")
        if not exists:
            all_ok = False
    
    # Check tokenizer files
    logger.info("\n🟡 TOKENIZER FILES (at least one needed):")
    tokenizer_found = False
    for filename, description in tokenizer_files.items():
        exists = os.path.exists(os.path.join(model_dir, filename))
        if exists:
            tokenizer_found = True
        status = "✅" if exists else "⚪"
        logger.info(f"  {status} {filename:30s} - {description}")
    
    if not tokenizer_found:
        logger.error("  ❌ NO TOKENIZER FILES FOUND - This will cause loading to fail!")
        all_ok = False
    
    # Check optional files
    logger.info("\n🟢 OPTIONAL FILES (recommended):")
    for filename, description in optional_files.items():
        exists = os.path.exists(os.path.join(model_dir, filename))
        status = "✅" if exists else "⚪"
        logger.info(f"  {status} {filename:30s} - {description}")
    
    logger.info("="*70)
    return all_ok


def validate_config_json(model_dir):
    """Validate the config.json file"""
    logger.info("\n" + "="*70)
    logger.info("⚙️  CONFIG.JSON VALIDATION")
    logger.info("="*70)
    
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(config_path):
        logger.error("❌ config.json not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("✅ config.json is valid JSON")
        logger.info("\nKey parameters:")
        
        important_keys = [
            "model_type",
            "num_layers", 
            "num_heads",
            "vocab_size",
            "quantization"
        ]
        
        for key in important_keys:
            value = config.get(key, "NOT FOUND")
            warning = ""
            
            # Check for issues
            if key == "quantization" and value not in ["float16", "float32", "int8", "int8_float16"]:
                warning = " ⚠️  Unexpected quantization type"
            
            logger.info(f"  {key:20s}: {value}{warning}")
        
        # Check model size estimate
        if "num_layers" in config:
            layers = config["num_layers"]
            if layers >= 32:
                logger.info(f"\n💡 This is a LARGE model ({layers} layers)")
                logger.info("   Requires significant GPU memory (8-16GB)")
            elif layers >= 24:
                logger.info(f"\n💡 This is a MEDIUM model ({layers} layers)")
                logger.info("   Requires moderate GPU memory (4-8GB)")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ config.json is corrupted: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error reading config.json: {e}")
        return False


def check_model_bin(model_dir):
    """Check model.bin file"""
    logger.info("\n" + "="*70)
    logger.info("🔍 MODEL.BIN ANALYSIS")
    logger.info("="*70)
    
    model_path = os.path.join(model_dir, "model.bin")
    
    if not os.path.exists(model_path):
        logger.error("❌ model.bin not found!")
        return False
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    logger.info(f"Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    
    # Expected sizes for different models
    expected_sizes = {
        "tiny": (30, 80),      # 30-80 MB
        "base": (100, 200),    # 100-200 MB
        "small": (400, 600),   # 400-600 MB
        "medium": (1300, 1700), # 1.3-1.7 GB
        "large-v2": (2800, 3200), # 2.8-3.2 GB
        "large-v3": (2800, 3200), # 2.8-3.2 GB
    }
    
    # Determine likely model size
    model_type = "unknown"
    for mtype, (min_size, max_size) in expected_sizes.items():
        if min_size <= size_mb <= max_size:
            model_type = mtype
            break
    
    if model_type != "unknown":
        logger.info(f"✅ Size matches '{model_type}' model")
    else:
        logger.warning("⚠️  Size doesn't match any known model type")
        logger.warning("   Expected sizes:")
        for mtype, (min_size, max_size) in expected_sizes.items():
            logger.warning(f"     {mtype:10s}: {min_size}-{max_size} MB")
    
    # Try to read first few bytes to verify it's not corrupted
    try:
        with open(model_path, 'rb') as f:
            header = f.read(16)
            if len(header) < 16:
                logger.error("❌ File is too small or corrupted")
                return False
        logger.info("✅ File header looks valid")
    except Exception as e:
        logger.error(f"❌ Cannot read file: {e}")
        return False
    
    return True


def test_load_with_faster_whisper(model_dir):
    """Try to actually load the model"""
    logger.info("\n" + "="*70)
    logger.info("🧪 LOADING TEST WITH FASTER-WHISPER")
    logger.info("="*70)
    
    try:
        from faster_whisper import WhisperModel
        import numpy as np
        
        logger.info("Attempting to load model...")
        logger.info("Device: CPU (safer for testing)")
        
        model = WhisperModel(
            model_dir,
            device="cpu",
            compute_type="float32",
            cpu_threads=2,
            num_workers=1,
            local_files_only=True
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Test transcription
        logger.info("\nTesting transcription with dummy audio...")
        test_audio = np.random.randn(16000).astype(np.float32)
        
        segments, info = model.transcribe(
            test_audio,
            beam_size=1,
            language="en"
        )
        
        # Consume segments
        segment_list = list(segments)
        
        logger.info(f"✅ Transcription test passed!")
        logger.info(f"   Language: {info.language}")
        logger.info(f"   Segments: {len(segment_list)}")
        
        return True
        
    except ImportError:
        logger.error("❌ faster-whisper not installed")
        logger.error("   Install with: pip install faster-whisper")
        return False
    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Run full diagnostic"""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("🏥 WHISPER MODEL DIAGNOSTIC TOOL")
    logger.info("=" * 70)
    
    # Get model directory
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        # Default to fine-tuned model path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(
            script_dir,
            "../models/faster-whisper-finetuned-topupwallet"
        )
    
    model_dir = os.path.abspath(model_dir)
    
    logger.info(f"\n📂 Analyzing model directory:")
    logger.info(f"   {model_dir}")
    
    if not os.path.exists(model_dir):
        logger.error("\n❌ DIRECTORY NOT FOUND!")
        logger.error("   Please provide the correct path")
        logger.error("\nUsage:")
        logger.error("   python diagnose_model.py [model_directory]")
        sys.exit(1)
    
    if not os.path.isdir(model_dir):
        logger.error("\n❌ PATH IS NOT A DIRECTORY!")
        sys.exit(1)
    
    # Run all checks
    checks_passed = []
    
    # Check 1: File sizes
    check_file_sizes(model_dir)
    
    # Check 2: Required files
    files_ok = check_required_files(model_dir)
    checks_passed.append(("Required files", files_ok))
    
    # Check 3: Config validation
    config_ok = validate_config_json(model_dir)
    checks_passed.append(("Config validation", config_ok))
    
    # Check 4: Model.bin validation
    model_bin_ok = check_model_bin(model_dir)
    checks_passed.append(("Model.bin", model_bin_ok))
    
    # Check 5: Loading test
    loading_ok = test_load_with_faster_whisper(model_dir)
    checks_passed.append(("Loading test", loading_ok))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 DIAGNOSTIC SUMMARY")
    logger.info("=" * 70)
    
    all_passed = True
    for check_name, passed in checks_passed:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} - {check_name}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 70)
    
    if all_passed:
        logger.info("\n🎉 ALL CHECKS PASSED!")
        logger.info("Your model should work fine with faster-whisper")
    else:
        logger.info("\n⚠️  SOME CHECKS FAILED")
        logger.info("\n💡 RECOMMENDED ACTIONS:")
        logger.info("1. Re-run the conversion script:")
        logger.info("   python convert_whisper_to_ctranslate2.py")
        logger.info("")
        logger.info("2. Verify your source model is complete")
        logger.info("")
        logger.info("3. Check conversion logs for errors")
        logger.info("")
        logger.info("4. If conversion keeps failing, try:")
        logger.info("   - Using a smaller base model (base/small instead of large)")
        logger.info("   - Converting with int8 quantization")
        logger.info("   - Using the base model instead of fine-tuned")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()