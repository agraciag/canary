from nemo.collections.asr.models import EncDecMultiTaskModel
import os
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Create a directory for test audio files
os.makedirs("/workspace/test_audio", exist_ok=True)

# Download a test audio file if not exists
test_audio_path = "/workspace/test_audio/test_sample.flac"
if not os.path.exists(test_audio_path):
    import urllib.request
    print("Downloading test audio file...")
    urllib.request.urlretrieve(
        "https://cdn-media.huggingface.co/speech_samples/sample1.flac", 
        test_audio_path
    )
    print(f"Test audio file downloaded to {test_audio_path}")

# Load model
print("Loading Canary-1B model...")
try:
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
    print("Model loaded successfully!")

    # Update decode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # Transcribe audio (English ASR by default)
    print("\nTranscribing English audio...")
    predicted_text = canary_model.transcribe(
        audio=[test_audio_path],
        batch_size=1
    )
    print(f"Transcription result: {predicted_text}")

    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error loading or using model: {str(e)}")