import torch
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load model on GPU
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Text you want to speak
text = "i am roshan. how can i help you?"

# Path to your reference clip (10s WAV)
reference_path = "recording.m4a"

# Generate audio using the reference voice
wav = model.generate(text, audio_prompt_path=reference_path)

# Convert tensor to numpy
wav_np = wav.squeeze().cpu().numpy()

# Save output
sf.write("test-turbo-with-ref.wav", wav_np, model.sr)

print("Audio generated using reference clip: test-turbo-with-ref.wav")
