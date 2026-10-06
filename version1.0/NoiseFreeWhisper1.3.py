import numpy as np
import torch
import whisper
import threading
import queue
from mic_noise_suppressor import MicNoiseSuppressor
import time

SAMPLE_RATE = 16000
SILENCE_SECONDS = 2
SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_SECONDS
SILENCE_THRESHOLD = 0.01  # energy threshold for silence

audio_queue = queue.Queue()

def is_silent(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < SILENCE_THRESHOLD

def has_voice(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms >= SILENCE_THRESHOLD


def audio_capture_thread(ns):
    buffer = []
    buffer_samples = 0
    silence_samples = 0

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

            if has_voice(audio):
                audio_queue.put(audio)
            # else: drop silent chunk

            buffer = []
            buffer_samples = 0
            silence_samples = 0



def transcription_thread(model):
    while True:
        audio = audio_queue.get()  # blocking wait

        result = model.transcribe(
            audio,
            fp16=True,
            verbose=False,
            language="en",
            no_speech_threshold=0.9,
            logprob_threshold=-1.0,
            temperature=0.0
        )


        print("\n--- TRANSCRIPTION ---")
        print(result["text"])
        audio_queue.task_done()


def main():
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Whisper will run on CPU.")

    print("Using GPU:", torch.cuda.get_device_name(0))

    # Load Whisper medium model on GPU
    model = whisper.load_model("large.en", device="cuda")

    # Start noise suppression mic pipeline
    ns = MicNoiseSuppressor()
    ns.start()
    print("Mic noise suppression started... Speak into the mic.")

    # Start threads
    t1 = threading.Thread(target=audio_capture_thread, args=(ns,), daemon=True)
    t2 = threading.Thread(target=transcription_thread, args=(model,), daemon=True)

    t1.start()
    t2.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping…")
        ns.stop()


if __name__ == "__main__":
    main()
