import numpy as np
import torch
import whisper
from mic_noise_suppressor import MicNoiseSuppressor

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5  # how long each audio chunk for Whisper
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

def main():
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Whisper will run on CPU.")

    print("Using GPU:", torch.cuda.get_device_name(0))

    # Load Whisper medium model on GPU
    model = whisper.load_model("medium.en", device="cuda")

    # Start noise suppression mic pipeline
    ns = MicNoiseSuppressor()
    ns.start()
    print("Mic noise suppression started... Speak into the mic.")

    buffer = []
    buffer_samples = 0

    try:
        while True:
            frame = ns.get_frame(timeout=1.0)
            if frame is None:
                continue

            buffer.append(frame)
            buffer_samples += len(frame)

            if buffer_samples >= CHUNK_SAMPLES:
                # Concatenate frames into one chunk
                audio = np.concatenate(buffer)[:CHUNK_SAMPLES]

                # Whisper expects float32, 16kHz
                result = model.transcribe(
                    audio,
                    fp16=True,
                    verbose=False,
                    language="en"
                )

                # Print transcription
                print("\n--- TRANSCRIPTION ---")
                print(result["text"])

                # Reset buffer for next chunk
                buffer = []
                buffer_samples = 0

    except KeyboardInterrupt:
        print("Stopping…")
        ns.stop()

if __name__ == "__main__":
    main()
