#!/usr/bin/env python3

import os
import argparse
from nemo.collections.asr.models import EncDecMultiTaskModel

def transcribe_audio(audio_path, source_lang="en", target_lang="en", task="asr"):
    """Transcribe audio file using Canary model"""
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File not found - {audio_path}")
        return
    
    # Load model
    print(f"Loading Canary-1B model...")
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    
    # Update decode params
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    
    if task == "asr" and source_lang == target_lang:
        # Simple transcription
        print(f"\nTranscribing audio in {source_lang}...")
        result = model.transcribe(audio=[audio_path], batch_size=1)
        print("\nTranscription result:")
        print(result[0])
        return result[0]
    else:
        # Create manifest file for translation
        print(f"\nTranslating from {source_lang} to {target_lang}...")
        manifest_path = "/workspace/temp_manifest.json"
        
        import json
        with open(manifest_path, 'w') as f:
            entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": 1000,  # placeholder
                "taskname": "s2t_translation" if task == "translation" else "asr",
                "source_lang": source_lang,
                "target_lang": target_lang,
                "pnc": "yes",
                "answer": "na"
            }
            f.write(json.dumps(entry))
        
        # Transcribe with manifest
        result = model.transcribe(manifest_path, batch_size=1)
        print("\nTranslation result:")
        print(result[0])
        
        # Clean up
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            
        return result[0]

def main():
    parser = argparse.ArgumentParser(description="Simple audio transcription/translation")
    parser.add_argument("audio", type=str, help="Path to audio file")
    parser.add_argument("--task", choices=["asr", "translation"], default="asr",
                        help="Task to perform (asr or translation)")
    parser.add_argument("--source-lang", choices=["en", "de", "es", "fr"], default="en",
                        help="Source language")
    parser.add_argument("--target-lang", choices=["en", "de", "es", "fr"], default="en",
                        help="Target language")
    
    args = parser.parse_args()
    
    transcribe_audio(args.audio, args.source_lang, args.target_lang, args.task)

if __name__ == "__main__":
    main()