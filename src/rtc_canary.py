import sounddevice as sd
import numpy as np
import wave
import tempfile
from nemo.collections.asr.models import EncDecMultiTaskModel

def record_audio(duration=5, fs=16000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

def save_wav(audio_data, filename, fs=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

def main():
    print("Loading Canary model...")
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)

    print("Starting transcription (Ctrl+C to stop)")
    while True:
        try:
            input("Press Enter to record 5 seconds...")
            print("Recording...")
            audio_data = record_audio()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                save_wav(audio_data, tmp_file.name)
                transcription = model.transcribe([tmp_file.name])
                print("Transcription:", transcription[0])
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()