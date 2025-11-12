"""
Convert Fine-tuned Whisper Model to CTranslate2 Format for faster-whisper
Fixed: CUDA device mismatch error by forcing CPU-only conversion

This script converts your fine-tuned Whisper model from HuggingFace format
to CTranslate2 format for use with faster-whisper library.

Requirements:
    pip install ctranslate2 transformers torch
"""

import os
import sys
import shutil
import logging

# ============================================================================
# CRITICAL: DISABLE CUDA BEFORE ANY IMPORTS
# This prevents the "cuda:0 and cpu device" error during conversion
# ============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Now safe to import torch and other libraries
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_cuda_disabled():
    """
    Verify CUDA is properly disabled to prevent device mismatch errors.
    Returns True if CUDA is disabled, False otherwise.
    """
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        logger.error("=" * 70)
        logger.error("CRITICAL ERROR: CUDA is still enabled!")
        logger.error("=" * 70)
        logger.error("The conversion will fail with device mismatch errors.")
        logger.error("")
        logger.error("To fix this, run the script in a fresh terminal:")
        logger.error("  Windows CMD:")
        logger.error("    set CUDA_VISIBLE_DEVICES=-1")
        logger.error("    python convert_whisper_to_ctranslate2.py")
        logger.error("")
        logger.error("  Windows PowerShell:")
        logger.error("    $env:CUDA_VISIBLE_DEVICES='-1'")
        logger.error("    python convert_whisper_to_ctranslate2.py")
        logger.error("")
        logger.error("  Linux/Mac:")
        logger.error("    export CUDA_VISIBLE_DEVICES=-1")
        logger.error("    python convert_whisper_to_ctranslate2.py")
        logger.error("=" * 70)
        return False
    
    logger.info("✅ CUDA successfully disabled - conversion will use CPU only")
    return True


def convert_finetuned_whisper_to_ctranslate2(
    source_model_path: str,
    output_model_path: str,
    quantization: str = "float16"
):
    """
    Convert fine-tuned Whisper model to CTranslate2 format using CPU only.
    
    Args:
        source_model_path: Path to your fine-tuned model directory
        output_model_path: Path where CTranslate2 model will be saved
        quantization: Quantization type - options: "int8", "int8_float16", "float16", "float32"
                     Note: Even with "float16", conversion happens on CPU
    """
    try:
        from ctranslate2.converters import TransformersConverter
        logger.info("✅ CTranslate2 imported successfully")
    except ImportError:
        logger.error("❌ CTranslate2 not found. Install with: pip install ctranslate2")
        return False
    
    try:
        logger.info("=" * 70)
        logger.info("STARTING CPU-ONLY CONVERSION")
        logger.info("=" * 70)
        logger.info(f"🔄 Source model: {source_model_path}")
        logger.info(f"📁 Output directory: {output_model_path}")
        logger.info(f"⚙️  Quantization: {quantization}")
        logger.info(f"💻 Device: CPU (forced)")
        logger.info("")
        logger.info("⏳ This may take 5-10 minutes depending on your system...")
        logger.info("=" * 70)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_model_path, exist_ok=True)
        
        # Create converter - it will automatically use CPU since CUDA is disabled
        logger.info("🔧 Initializing converter...")
        converter = TransformersConverter(
            model_name_or_path=source_model_path
        )
        
        # Convert the model
        logger.info("🔄 Converting model (please wait)...")
        converter.convert(
            output_dir=output_model_path,
            quantization=quantization,
            force=True  # Overwrite if exists
        )
        
        logger.info("=" * 70)
        logger.info("✅ MODEL CONVERSION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"📊 Converted model saved to: {output_model_path}")
        logger.info("")
        
        # Copy necessary config files from source
        config_files = [
            "preprocessor_config.json",
            "tokenizer.json", 
            "vocab.json",
            "vocabulary.json",
            "merges.txt",
            "normalizer.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "added_tokens.json"
        ]
        
        logger.info("📋 Copying configuration files...")
        copied_files = []
        for file_name in config_files:
            src_file = os.path.join(source_model_path, file_name)
            dst_file = os.path.join(output_model_path, file_name)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                copied_files.append(file_name)
                logger.info(f"   ✓ Copied {file_name}")
        
        if not copied_files:
            logger.warning("   ⚠ No additional config files found to copy")
        
        logger.info("")
        
        # Verify the conversion
        logger.info("🔍 Verifying converted model...")
        required_files = ["model.bin", "config.json"]
        missing_files = []
        
        for file_name in required_files:
            file_path = os.path.join(output_model_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
            else:
                # Show file size
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"   ✓ {file_name:30s} ({size_mb:>8.2f} MB)")
        
        if missing_files:
            logger.error(f"❌ Conversion incomplete. Missing files: {missing_files}")
            return False
        
        # List all files in output directory
        logger.info("")
        logger.info("📦 All files in output directory:")
        for file_name in sorted(os.listdir(output_model_path)):
            file_path = os.path.join(output_model_path, file_name)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"   • {file_name:30s} ({size_mb:>8.2f} MB)")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎉 CONVERSION COMPLETE!")
        logger.info("=" * 70)
        return True
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("❌ CONVERSION FAILED")
        logger.error("=" * 70)
        logger.exception(f"Error during conversion: {e}")
        logger.error("")
        logger.error("💡 Troubleshooting tips:")
        logger.error("1. Ensure source model files are complete and not corrupted")
        logger.error("2. Check you have enough disk space (need ~2x model size)")
        logger.error("3. Try running in a fresh Python session")
        logger.error("4. Verify ctranslate2 version: pip show ctranslate2")
        logger.error("=" * 70)
        return False


def print_usage_instructions(output_path: str):
    """
    Print instructions for using the converted model.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("📚 HOW TO USE YOUR CONVERTED MODEL")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Option 1: With faster-whisper (Recommended)")
    logger.info("-" * 70)
    logger.info("from faster_whisper import WhisperModel")
    logger.info("")
    logger.info("model = WhisperModel(")
    logger.info(f'    r"{os.path.abspath(output_path)}",')
    logger.info('    device="cuda",        # or "cpu"')
    logger.info('    compute_type="float16" # or "int8" for CPU')
    logger.info(")")
    logger.info("")
    logger.info('segments, info = model.transcribe("audio.wav", language="en")')
    logger.info("for segment in segments:")
    logger.info('    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")')
    logger.info("")
    logger.info("-" * 70)
    logger.info("Option 2: Update your existing application")
    logger.info("-" * 70)
    logger.info("Update the LOCAL_MODEL_DIR in your loaders.py:")
    logger.info("")
    logger.info("LOCAL_MODEL_DIR = os.path.join(")
    logger.info("    os.path.dirname(__file__),")
    logger.info('    "../models",')
    logger.info(f'    "{os.path.basename(output_path)}"')
    logger.info(")")
    logger.info("")
    logger.info("=" * 70)


def main():
    """
    Main conversion script with improved error handling
    """
    # ========================================================================
    # CONFIGURATION - Update these paths for your setup
    # ========================================================================
    FINETUNED_MODEL_PATH = "whisper-finetuned-topupwallet"
    OUTPUT_MODEL_PATH = "faster-whisper-finetuned-topupwallet"
    
    # Quantization options:
    # - "float16": Best quality (recommended, works on CPU during conversion)
    # - "int8": Smaller size, faster on CPU during inference
    # - "int8_float16": Balanced option
    QUANTIZATION = "float16"
    
    logger.info("=" * 70)
    logger.info("🚀 WHISPER MODEL CONVERSION TOOL")
    logger.info("=" * 70)
    logger.info(f"Source: {FINETUNED_MODEL_PATH}")
    logger.info(f"Output: {OUTPUT_MODEL_PATH}")
    logger.info(f"Quantization: {QUANTIZATION}")
    logger.info("=" * 70)
    logger.info("")
    
    # Step 1: Verify CUDA is disabled
    logger.info("Step 1: Verifying CUDA is disabled...")
    if not verify_cuda_disabled():
        logger.error("❌ Please disable CUDA and try again (see instructions above)")
        sys.exit(1)
    logger.info("")
    
    # Step 2: Verify source model exists
    logger.info("Step 2: Verifying source model...")
    if not os.path.exists(FINETUNED_MODEL_PATH):
        logger.error(f"❌ Source model not found at: {FINETUNED_MODEL_PATH}")
        logger.error("Please check the path and try again.")
        sys.exit(1)
    
    # Check for required files in source
    required_source_files = ["model.safetensors", "config.json"]
    missing_source_files = []
    
    for file_name in required_source_files:
        if not os.path.exists(os.path.join(FINETUNED_MODEL_PATH, file_name)):
            missing_source_files.append(file_name)
    
    if missing_source_files:
        logger.error(f"❌ Source model incomplete. Missing: {missing_source_files}")
        logger.error("Please ensure you have a complete fine-tuned model.")
        sys.exit(1)
    
    logger.info("✅ Source model validation passed")
    logger.info("")
    
    # Step 3: Perform conversion
    logger.info("Step 3: Converting model...")
    logger.info("")
    
    success = convert_finetuned_whisper_to_ctranslate2(
        source_model_path=FINETUNED_MODEL_PATH,
        output_model_path=OUTPUT_MODEL_PATH,
        quantization=QUANTIZATION
    )
    
    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ SUCCESS! YOUR MODEL IS READY")
        logger.info("=" * 70)
        
        # Print usage instructions
        print_usage_instructions(OUTPUT_MODEL_PATH)
        
        logger.info("📝 Next Steps:")
        logger.info("1. Test the model with a sample audio file")
        logger.info("2. Update your application to use the new model path")
        logger.info("3. Compare accuracy with the base model")
        logger.info("4. Monitor performance in your application")
        logger.info("")
        logger.info("💡 Pro Tip: Keep your original fine-tuned model as backup!")
        logger.info("=" * 70)
    else:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ CONVERSION FAILED")
        logger.error("=" * 70)
        logger.error("Please check the errors above and try again.")
        logger.error("If the issue persists, consider:")
        logger.error("1. Using the model directly with transformers library")
        logger.error("2. Re-training with a different base model")
        logger.error("3. Checking ctranslate2 GitHub issues for similar problems")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()