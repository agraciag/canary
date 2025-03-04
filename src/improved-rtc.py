#!/usr/bin/env python3

import os
import sys
import argparse
import time
import json
import datetime
import threading
import queue
import numpy as np
import sounddevice as sd
from pathlib import Path
import soundfile as sf
from nemo.collections.asr.models import EncDecMultiTaskModel
import curses
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

class RealTimeCanary:
    def __init__(self, device=None, samplerate=16000, channels=1, 
                 source_lang="en", target_lang="en", task="asr", 
                 pnc="yes", beam_size=1, buffer_size=3):
        """
        Initialize RealTimeCanary
        
        Args:
            device: Input device (sounddevice)
            samplerate: Audio sampling rate (should be 16000 for Canary)
            channels: Number of channels (Canary expects mono)
            source_lang: Source language (en, de, es, fr)
            target_lang: Target language (en, de, es, fr)
            task: Task to perform (asr or translation)
            pnc: Include punctuation and capitalization (yes/no)
            beam_size: Beam size for decoding
            buffer_size: Size of audio buffer in seconds
        """
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.task = task
        self.taskname = "asr" if task == "asr" else "s2t_translation"
        self.pnc = pnc
        self.beam_size = beam_size
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.transcript_buffer = []
        self.current_transcript = ""
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        
        # Create necessary directories
        self.transcript_dir = "/workspace/transcripts"
        self.temp_dir = "/workspace/temp_audio"
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load Canary model
        self.console.print("[bold blue]Loading Canary-1B model...[/bold blue]")
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        
        # Update decode params
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        self.model.change_decoding_strategy(decode_cfg)
        self.console.print("[bold green]Model loaded successfully![/bold green]")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to capture audio"""
        if status:
            print(status, file=sys.stderr)
        # Add the audio data to the queue
        self.audio_queue.put(indata.copy())
    
    def create_manifest(self, audio_file):
        """Create a manifest file for the audio file"""
        manifest_path = f"{self.temp_dir}/manifest_{self.session_id}_{int(time.time())}.json"
        
        entry = {
            "audio_filepath": os.path.abspath(audio_file),
            "duration": self.buffer_size,
            "taskname": self.taskname,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pnc": self.pnc,
            "answer": "na"
        }
        
        with open(manifest_path, 'w') as f:
            f.write(json.dumps(entry))
            
        return manifest_path
    
    def process_audio(self):
        """Process audio chunks from the queue and transcribe"""
        buffer = np.array([]).reshape(0, self.channels)
        buffer_samples = int(self.samplerate * self.buffer_size)
        overlap_samples = int(buffer_samples * 0.25)  # 25% overlap for context
        chunk_index = 0
        
        # Setup rich display
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        def get_header():
            mode = f"[{self.source_lang}->{self.target_lang}]" if self.task == "translation" else f"[{self.source_lang}]"
            return Panel(
                f"Real-Time Canary {self.task.upper()} {mode} - Buffer: {self.buffer_size}s - Press Ctrl+C to stop",
                style="bold blue on black"
            )
            
        def get_footer():
            return Panel(
                f"Session: {self.session_id} | Chunks processed: {chunk_index}",
                style="bold white on black"
            )
            
        def get_main():
            lines = textwrap.wrap(self.current_transcript, width=80) if self.current_transcript else ["Listening..."]
            content = "\n".join(lines)
            return Panel(content, title="Transcript", border_style="green")
        
        with Live(layout, refresh_per_second=4) as live:
            layout["header"].update(get_header())
            layout["footer"].update(get_footer())
            layout["main"].update(get_main())
            
            while not self.stop_event.is_set():
                try:
                    # Get audio chunk from queue with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                    buffer = np.vstack((buffer, chunk))
                    
                    # Process when buffer is full
                    if len(buffer) >= buffer_samples:
                        # Save audio to temporary file
                        audio_file = f"{self.temp_dir}/chunk_{self.session_id}_{chunk_index}.wav"
                        sf.write(audio_file, buffer, self.samplerate)
                        
                        # Create manifest for processing
                        manifest_path = self.create_manifest(audio_file)
                        
                        # Process with manifest
                        result = self.model.transcribe(manifest_path, batch_size=1)
                        
                        if result and len(result) > 0:
                            # Add to transcript buffer
                            self.transcript_buffer.append(result[0])
                            
                            # Update current transcript (last 3 chunks)
                            self.current_transcript = " ".join(self.transcript_buffer[-3:])
                            layout["main"].update(get_main())
                            layout["footer"].update(get_footer())
                        
                        # Reset buffer with overlap for context
                        buffer = buffer[-overlap_samples:] if overlap_samples > 0 else np.array([]).reshape(0, self.channels)
                        
                        # Clean up temporary files
                        os.remove(audio_file)
                        os.remove(manifest_path)
                        
                        chunk_index += 1
                
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[bold red]Error processing audio: {str(e)}[/bold red]")
        
        return self.transcript_buffer
    
    def save_transcript(self, transcript_buffer):
        """Save complete transcript to file"""
        if not transcript_buffer:
            return
            
        # Create filename based on task
        if self.task == "asr":
            filename = f"{self.transcript_dir}/realtime_{self.source_lang}_transcription_{self.session_id}.txt"
        else:
            filename = f"{self.transcript_dir}/realtime_{self.source_lang}_to_{self.target_lang}_{self.session_id}.txt"
        
        # Write transcript to file
        with open(filename, 'w') as f:
            f.write("\n".join(transcript_buffer))
            
        self.console.print(f"\n[bold green]Transcript saved to {filename}[/bold green]")
        return filename
    
    def run(self):
        """Run real-time transcription"""
        try:
            # Start audio capture thread
            with sd.InputStream(device=self.device, channels=self.channels,
                               samplerate=self.samplerate, callback=self.audio_callback):
                
                # Process audio in main thread
                transcript_buffer = self.process_audio()
                
                # Save complete transcript
                filename = self.save_transcript(transcript_buffer)
                
                # Print summary
                if filename:
                    word_count = sum(len(line.split()) for line in transcript_buffer)
                    self.console.print(f"[bold]Session Summary:[/bold]")
                    self.console.print(f"- Duration: {len(transcript_buffer) * self.buffer_size:.1f} seconds (approx)")
                    self.console.print(f"- Words transcribed: {word_count}")
                    self.console.print(f"- Chunks processed: {len(transcript_buffer)}")
                
        except KeyboardInterrupt:
            self.stop_event.set()
            self.console.print("\n[bold yellow]Stopping...[/bold yellow]")
        except Exception as e:
            self.console.print(f"[bold red]Error: {str(e)}[/bold red]")

def list_devices():
    """List available audio devices"""
    console = Console()
    console.print("[bold]Available audio devices:[/bold]")
    devices = sd.query_devices()
    table = []
    for i, device in enumerate(devices):
        console.print(f"[bold]{i}:[/bold] {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")

def main():
    parser = argparse.ArgumentParser(description="Real-time Canary ASR/Translation")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--task", choices=["asr", "translation"], default="asr", 
                        help="Task to perform (asr or translation)")
    parser.add_argument("--source-lang", choices=["en", "de", "es", "fr"], default="en",
                        help="Source language")
    parser.add_argument("--target-lang", choices=["en", "de", "es", "fr"], default="en",
                        help="Target language")
    parser.add_argument("--pnc", choices=["yes", "no"], default="yes",
                        help="Include punctuation and capitalization")
    parser.add_argument("--buffer-size", type=float, default=3.0, 
                        help="Audio buffer size in seconds")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    # Check if task and languages make sense
    if args.task == "asr" and args.source_lang != args.target_lang:
        console = Console()
        console.print(f"[bold yellow]Warning: For ASR, source and target languages should be the same. Setting target_lang to {args.source_lang}[/bold yellow]")
        args.target_lang = args.source_lang
    
    # Create and run real-time transcription
    rtc = RealTimeCanary(
        device=args.device,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        task=args.task,
        pnc=args.pnc,
        buffer_size=args.buffer_size,
        beam_size=args.beam_size
    )
    
    rtc.run()

if __name__ == "__main__":
    main()