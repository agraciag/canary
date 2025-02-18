import numpy as np
import pyaudio
import wave
import threading
import queue
import tempfile
from nemo.collections.asr.models import EncDecMultiTaskModel

class AudioTranscriber:
    def __init__(self, model_name='nvidia/canary-1b'):
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 3
        self.audio_queue = queue.Queue()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Load Canary model
        self.model = EncDecMultiTaskModel.from_pretrained(model_name)
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        self.model.change_decoding_strategy(decode_cfg)
        
        self.is_recording = False
        
    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.transcription_thread = threading.Thread(target=self._process_audio)
        
        self.recording_thread.start()
        self.transcription_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        self.recording_thread.join()
        self.transcription_thread.join()
        
    def _record_audio(self):
        stream = self.p.open(format=self.format,
                           channels=self.channels,
                           rate=self.rate,
                           input=True,
                           frames_per_buffer=self.chunk)
        
        while self.is_recording:
            frames = []
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk)
                frames.append(data)
            
            if frames:
                self.audio_queue.put(frames)
        
        stream.stop_stream()
        stream.close()
        
    def _process_audio(self):
        while self.is_recording or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                frames = self.audio_queue.get()
                
                # Save temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                    with wave.open(temp_wav.name, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.p.get_sample_size(self.format))
                        wf.setframerate(self.rate)
                        wf.writeframes(b''.join(frames))
                    
                    # Transcribe using Canary
                    try:
                        transcription = self.model.transcribe([temp_wav.name])
                        if transcription:
                            print("Transcription:", transcription[0])
                    except Exception as e:
                        print(f"Error in transcription: {e}")
        
    def cleanup(self):
        self.p.terminate()

if __name__ == "__main__":
    transcriber = AudioTranscriber()
    print("Starting recording (press Ctrl+C to stop)...")
    try:
        transcriber.start_recording()
        input("Press Enter to stop recording...")
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop_recording()
        transcriber.cleanup()