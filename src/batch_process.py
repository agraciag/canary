#!/usr/bin/env python3

import os
import argparse
import json
import datetime
from pathlib import Path
from nemo.collections.asr.models import EncDecMultiTaskModel

def process_directory(audio_dir, output_dir, task, source_lang, target_lang, pnc, batch_size, beam_size):
    """
    Process all audio files in a directory
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save transcriptions/translations
        task: Task to perform (asr or translation)
        source_lang: Source language
        target_lang: Target language
        pnc: Include punctuation and capitalization
        batch_size: Batch size for processing
        beam_size: Beam size for decoding
    """
    # Find all audio files in directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Canary model
    print("Loading Canary-1B model...")
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    
    # Update decode params
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = beam_size
    model.change_decoding_strategy(decode_cfg)
    print("Model loaded successfully!")
    
    # Create manifest for processing
    taskname = "asr" if task == "asr" else "s2t_translation"
    manifest_path = f"{output_dir}/batch_manifest.json"
    
    with open(manifest_path, 'w') as f:
        for audio_path in audio_files:
            entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": 1000,  # placeholder
                "taskname": taskname,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "pnc": pnc,
                "answer": "na"
            }
            f.write(json.dumps(entry) + '\n')
    
    # Process with manifest
    print(f"\nProcessing {len(audio_files)} files with batch size {batch_size}...")
    results = model.transcribe(manifest_path, batch_size=batch_size)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Single output file with all results
    all_results_file = f"{output_dir}/all_results_{timestamp}.txt"
    with open(all_results_file, 'w') as f:
        f.write(f"# Batch processing results - {timestamp}\n")
        f.write(f"# Task: {task}, Source: {source_lang}, Target: {target_lang}\n\n")
        
        for i, (path, text) in enumerate(zip(audio_files, results)):
            f.write(f"## File: {Path(path).name}\n")
            f.write(f"{text}\n\n")
    
    print(f"All results saved to {all_results_file}")
    
    # Individual files for each result
    for i, (path, text) in enumerate(zip(audio_files, results)):
        filename = Path(path).stem
        if task == "asr":
            output_file = f"{output_dir}/{filename}_{source_lang}_transcription.txt"
        else:
            output_file = f"{output_dir}/{filename}_{source_lang}_to_{target_lang}.txt"
            
        with open(output_file, 'w') as f:
            f.write(text)
        
        print(f"Saved result for {Path(path).name} to {output_file}")
    
    # Clean up manifest
    os.remove(manifest_path)

def main():
    parser = argparse.ArgumentParser(description="Batch process audio files with Canary")
    parser.add_argument("--audio-dir", "-a", type=str, required=True, 
                        help="Directory containing audio files to process")
    parser.add_argument("--output-dir", "-o", type=str, default="/workspace/transcripts",
                        help="Directory to save results")
    parser.add_argument("--task", "-t", choices=["asr", "translation"], default="asr", 
                        help="Task to perform (asr or translation)")
    parser.add_argument("--source-lang", "-s", choices=["en", "de", "es", "fr"], default="en",
                        help="Source language")
    parser.add_argument("--target-lang", "-tl", choices=["en", "de", "es", "fr"], default="en",
                        help="Target language")
    parser.add_argument("--pnc", choices=["yes", "no"], default="yes",
                        help="Include punctuation and capitalization")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding")
    
    args = parser.parse_args()
    
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        parser.error(f"Audio directory not found: {args.audio_dir}")
    
    # Check if task and languages make sense
    if args.task == "asr" and args.source_lang != args.target_lang:
        print(f"Warning: For ASR, source and target languages should be the same. Setting target_lang to {args.source_lang}")
        args.target_lang = args.source_lang
    
    # Process directory
    process_directory(
        args.audio_dir,
        args.output_dir,
        args.task,
        args.source_lang,
        args.target_lang,
        args.pnc,
        args.batch_size,
        args.beam_size
    )

if __name__ == "__main__":
    main()