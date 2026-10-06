import numpy as np
import torch
import whisper
from mic_noise_suppressor import MicNoiseSuppressor

SAMPLE_RATE = 16000
SILENCE_SECONDS = 2
SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_SECONDS
SILENCE_THRESHOLD = 0.01  # energy threshold for silence

def is_silent(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < SILENCE_THRESHOLD

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
    silence_samples = 0

    try:
        while True:
            frame = ns.get_frame(timeout=1.0)
            if frame is None:
                continue

            buffer.append(frame)
            buffer_samples += len(frame)

            if is_silent(frame):
                silence_samples += len(frame)
            else:
                silence_samples = 0

            # If 2 seconds of silence detected → finalize chunk
            if silence_samples >= SILENCE_SAMPLES and buffer_samples > 0:
                audio = np.concatenate(buffer)

                result = model.transcribe(
                    audio,
                    fp16=True,
                    verbose=False,
                    language="en"
                )

                print("\n--- TRANSCRIPTION ---")
                print(result["text"])

                buffer = []
                buffer_samples = 0
                silence_samples = 0

    except KeyboardInterrupt:
        print("Stopping…")
        ns.stop()

if __name__ == "__main__":
    main()
