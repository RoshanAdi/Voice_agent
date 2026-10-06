import whisper
import pyaudio
import numpy as np
import torch

# Configuration
MODEL_NAME = "medium.en"  # You mentioned OpenAI's OSS model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize standard Whisper
print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE}...")
model = whisper.load_model(MODEL_NAME, device=DEVICE)

# Audio Stream Settings
RATE = 16000        # Whisper is trained on 16kHz audio
CHUNK = 1024        # Number of frames per buffer
RECORD_SECONDS = 3  # Transcribe in 3-second intervals for interactivity

def record_and_transcribe():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n🎤 Microphone active. Speak now (Ctrl+C to stop)...")

    try:
        while True:
            frames = []
            # Buffer audio for the specified interval
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))

            # Convert to float32 and normalize to [-1, 1] as Whisper expects
            audio_data = np.concatenate(frames).astype(np.float32) / 32768.0

            # Transcribe the numpy array directly
            # Setting fp16=False if running on CPU to avoid warnings
            result = model.transcribe(audio_data, fp16=(DEVICE == "cuda"))
            
            text = result['text'].strip()
            if text:
                print(f"🗨️  You said: {text}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    record_and_transcribe()
