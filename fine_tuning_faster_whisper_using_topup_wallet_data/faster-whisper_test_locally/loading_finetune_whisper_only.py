import torch
import librosa
import soundfile as sf
from transformers import pipeline, WhisperProcessor
import os
from pathlib import Path
import numpy as np

def chunk_long_audio(audio_path, max_duration=30.0, overlap=1.0):
    """
    Split long audio into chunks for processing.
    Returns list of chunk paths.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    
    if audio_duration <= max_duration:
        return [audio_path]
    
    chunk_samples = int(max_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    chunk_idx = 0
    os.makedirs("chunks", exist_ok=True)
    
    for start in range(0, len(audio) - overlap_samples, step_samples):
        end = min(start + chunk_samples, len(audio))
        chunk_audio = audio[start:end]
        
        chunk_filename = f"chunk_{chunk_idx}_{Path(audio_path).stem}.wav"
        chunk_path = f"chunks/{chunk_filename}"
        sf.write(chunk_path, chunk_audio, sr)
        
        chunks.append(chunk_path)
        chunk_idx += 1
        
        if end >= len(audio):
            break
    
    print(f"✅ Split {audio_path} ({audio_duration:.1f}s) into {len(chunks)} chunks")
    return chunks

# -------------------------
# Load Model and Processor
# -------------------------
MODEL_PATH = "../models/whisper-large-v3-malaysian-merged"

# Load the processor manually (fix for missing processor)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

# Load pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    chunk_length_s=30,
    batch_size=4
)

# Set forced decoder IDs properly
pipe.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english", 
    task="transcribe"
)

# -------------------------
# Process Long Audio
# -------------------------

audio_file = "../audio_samples/gaurav.mp3"

# Split into chunks if longer than 30s
chunk_paths = chunk_long_audio(audio_file)

# Transcribe each chunk
transcriptions = []
for i, chunk_path in enumerate(chunk_paths):
    print(f"\n🎧 Transcribing chunk {i+1}/{len(chunk_paths)}...")
    result = pipe(
        chunk_path,
        generate_kwargs={"language": "english", "task": "transcribe"}
    )
    text = result["text"]
    transcriptions.append(text)
    print(f"Chunk {i+1}: {text}")

# Concatenate results
full_trans = " ".join(transcriptions)
print("\n📝 Full Transcription:")
print(full_trans)

# Optional cleanup
for chunk in chunk_paths:
    if "chunks/" in chunk:
        os.remove(chunk)
