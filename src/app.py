#!/usr/bin/env python3

import os
import json
import argparse
import datetime
from pathlib import Path
from nemo.collections.asr.models import EncDecMultiTaskModel

class CanaryASR:
    def __init__(self, beam_size=1):
        print("Loading Canary-1B model...")
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        
        # Update decode params
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        self.model.change_decoding_strategy(decode_cfg)
        print("Model loaded successfully!")
        
    def transcribe_audio(self, audio_paths, batch_size=1):
        """Transcribe list of audio files (English ASR)"""
        return self.model.transcribe(
            paths2audio_files=audio_paths,
            batch_size=batch_size
        )
    
    def process_with_manifest(self, manifest_path, batch_size=1):
        """Process audio according to manifest file specifications"""
        return self.model.transcribe(
            manifest_path,
            batch_size=batch_size
        )
    
    def create_manifest(self, audio_paths, output_path, task_configs):
        """
        Create a manifest file for processing audio files with specific configurations
        
        Args:
            audio_paths (list): List of paths to audio files
            output_path (str): Path to save the manifest file
            task_configs (list): List of dictionaries with task configurations
                                Each dict needs: taskname, source_lang, target_lang, pnc
        """
        manifest_data = []
        
        for audio_path in audio_paths:
            abs_path = os.path.abspath(audio_path)
            for config in task_configs:
                entry = {
                    "audio_filepath": abs_path,
                    "duration": 1000,  # placeholder
                    "taskname": config["taskname"],
                    "source_lang": config["source_lang"],
                    "target_lang": config["target_lang"],
                    "pnc": config["pnc"],
                    "answer": "na"
                }
                manifest_data.append(entry)
        
        # Write the manifest file
        with open(output_path, 'w') as f:
            for entry in manifest_data:
                f.write(json.dumps(entry) + '\n')
                
        return manifest_data

    def save_results(self, results, audio_paths, task, source_lang, target_lang):
        """Save results to transcripts directory"""
        os.makedirs("/workspace/transcripts", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, (path, text) in enumerate(zip(audio_paths, results)):
            filename = Path(path).stem
            if task == "asr":
                output_file = f"/workspace/transcripts/{filename}_{source_lang}_transcription_{timestamp}.txt"
            else:
                output_file = f"/workspace/transcripts/{filename}_{source_lang}_to_{target_lang}_{timestamp}.txt"
                
            with open(output_file, "w") as f:
                f.write(text)
            print(f"Result saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Canary ASR/Translation CLI")
    parser.add_argument("--audio", "-a", type=str, nargs="+", help="Path to audio file(s)")
    parser.add_argument("--task", "-t", choices=["asr", "translation"], default="asr", 
                        help="Task to perform (asr or translation)")
    parser.add_argument("--source-lang", "-s", choices=["en", "de", "es", "fr"], default="en",
                        help="Source language")
    parser.add_argument("--target-lang", "-tl", choices=["en", "de", "es", "fr"], default="en",
                        help="Target language")
    parser.add_argument("--pnc", choices=["yes", "no"], default="yes",
                        help="Include punctuation and capitalization")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding")
    parser.add_argument("--save", action="store_true", help="Save results to transcripts directory")
    
    args = parser.parse_args()
    
    if not args.audio:
        parser.error("Please provide at least one audio file path")
    
    # Check if files exist
    for audio_path in args.audio:
        if not os.path.exists(audio_path):
            parser.error(f"Audio file not found: {audio_path}")
    
    # Initialize the model
    asr = CanaryASR(beam_size=args.beam_size)
    
    if args.task == "asr" and args.source_lang == "en" and args.target_lang == "en":
        # Simple English ASR
        results = asr.transcribe_audio(args.audio, batch_size=args.batch_size)
        
        for i, (path, text) in enumerate(zip(args.audio, results)):
            print(f"\nAudio {i+1}: {Path(path).name}")
            print(f"Transcription: {text}")
    else:
        # Create manifest for specified task
        task_name = "asr" if args.task == "asr" else "s2t_translation"
        config = {
            "taskname": task_name,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "pnc": args.pnc
        }
        
        manifest_path = "/workspace/temp_manifest.json"
        asr.create_manifest(args.audio, manifest_path, [config])
        
        # Process with manifest
        results = asr.process_with_manifest(manifest_path, batch_size=args.batch_size)
        
        for i, (path, text) in enumerate(zip(args.audio, results)):
            print(f"\nAudio {i+1}: {Path(path).name}")
            print(f"Source language: {args.source_lang}")
            print(f"Target language: {args.target_lang}")
            if args.task == "asr":
                print(f"Transcription: {text}")
            else:
                print(f"Translation: {text}")
        
        # Clean up temp manifest
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
    
    # Save results if requested
    if args.save:
        asr.save_results(results, args.audio, args.task, args.source_lang, args.target_lang)

if __name__ == "__main__":
    main()