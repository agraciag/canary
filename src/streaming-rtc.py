#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
import datetime
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import socket
from nemo.collections.asr.models import EncDecMultiTaskModel
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
transcription_queue = queue.Queue()
stop_event = threading.Event()
current_session = None

class TranscriptionSession:
    def __init__(self, device=None, samplerate=16000, channels=1, 
                 source_lang="en", target_lang="en", task="asr", 
                 pnc="yes", beam_size=1, buffer_size=2):
        """Initialize a transcription session with Canary model"""
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
        self.transcript_buffer = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create necessary directories
        self.transcript_dir = "/workspace/transcripts"
        self.temp_dir = "/workspace/temp_audio"
        os.makedirs(self.transcript_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load model
        print(f"Loading Canary-1B model for {task} ({source_lang}->{target_lang})...")
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        
        # Update decode params
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        self.model.change_decoding_strategy(decode_cfg)
        print("Model loaded successfully!")
        
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
    
    def process_audio_thread(self):
        """Process audio chunks from the queue and transcribe"""
        buffer = np.array([]).reshape(0, self.channels)
        buffer_samples = int(self.samplerate * self.buffer_size)
        overlap_samples = int(buffer_samples * 0.15)  # 15% overlap for context
        chunk_index = 0
        
        while not stop_event.is_set():
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
                    start_time = time.time()
                    result = self.model.transcribe(manifest_path, batch_size=1)
                    end_time = time.time()
                    
                    if result and len(result) > 0:
                        # Add to transcript buffer
                        self.transcript_buffer.append(result[0])
                        
                        # Send to web UI
                        processing_time = end_time - start_time
                        transcription_data = {
                            'text': result[0],
                            'chunk_index': chunk_index,
                            'processing_time': f"{processing_time:.2f}s",
                            'source_lang': self.source_lang,
                            'target_lang': self.target_lang,
                            'task': self.task
                        }
                        transcription_queue.put(transcription_data)
                    
                    # Reset buffer with overlap for context
                    buffer = buffer[-overlap_samples:] if overlap_samples > 0 else np.array([]).reshape(0, self.channels)
                    
                    # Clean up temporary files
                    try:
                        os.remove(audio_file)
                        os.remove(manifest_path)
                    except:
                        pass
                    
                    chunk_index += 1
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
        
        # Save final transcript
        self.save_transcript()
    
    def save_transcript(self):
        """Save complete transcript to file"""
        if not self.transcript_buffer:
            return
            
        # Create filename based on task
        if self.task == "asr":
            filename = f"{self.transcript_dir}/realtime_{self.source_lang}_transcription_{self.session_id}.txt"
        else:
            filename = f"{self.transcript_dir}/realtime_{self.source_lang}_to_{self.target_lang}_{self.session_id}.txt"
        
        # Write transcript to file
        with open(filename, 'w') as f:
            f.write("\n".join(self.transcript_buffer))
            
        print(f"Transcript saved to {filename}")
        
        # Send to UI
        socketio.emit('transcript_saved', {
            'filename': filename,
            'count': len(self.transcript_buffer),
            'word_count': sum(len(line.split()) for line in self.transcript_buffer)
        })
        
        return filename
    
    def start(self):
        """Start the transcription session"""
        # Create and start processing thread
        processing_thread = threading.Thread(target=self.process_audio_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start audio capture thread
        self.stream = sd.InputStream(
            device=self.device, 
            channels=self.channels,
            samplerate=self.samplerate, 
            callback=self.audio_callback
        )
        self.stream.start()
        
        return processing_thread
    
    def stop(self):
        """Stop the transcription session"""
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        self.save_transcript()

# Define Socket.IO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    
@socketio.on('start_transcription')
def handle_start_transcription(data):
    global current_session, stop_event
    
    # Stop any existing session
    if current_session:
        stop_event.set()
        current_session.stop()
        time.sleep(0.5)
    
    # Reset stop event
    stop_event.clear()
    
    # Extract parameters
    device = data.get('device')
    if device == 'default':
        device = None
    else:
        device = int(device)
    
    task = data.get('task', 'asr')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'en')
    pnc = data.get('pnc', 'yes')
    buffer_size = float(data.get('buffer_size', 2.0))
    beam_size = int(data.get('beam_size', 1))
    
    # Create and start new session
    current_session = TranscriptionSession(
        device=device,
        source_lang=source_lang,
        target_lang=target_lang,
        task=task,
        pnc=pnc,
        buffer_size=buffer_size,
        beam_size=beam_size
    )
    
    current_session.start()
    
    # Start emitting transcriptions
    def send_transcriptions():
        while not stop_event.is_set():
            try:
                data = transcription_queue.get(timeout=0.5)
                socketio.emit('transcription', data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error emitting transcription: {str(e)}")
    
    threading.Thread(target=send_transcriptions, daemon=True).start()
    
    return {'status': 'started', 'session_id': current_session.session_id}
    
@socketio.on('stop_transcription')
def handle_stop_transcription():
    global current_session, stop_event
    if current_session:
        stop_event.set()
        current_session.stop()
        return {'status': 'stopped'}
    return {'status': 'no_session'}

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/devices')
def get_devices():
    devices = sd.query_devices()
    device_list = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_list.append({
                'id': i,
                'name': device['name'],
                'inputs': device['max_input_channels'],
                'samplerate': device['default_samplerate']
            })
    
    return jsonify({'devices': device_list})

def get_ip_address():
    """Get the current machine's IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def create_templates():
    """Create template directory and HTML file"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    