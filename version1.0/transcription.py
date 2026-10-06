from faster_whisper import WhisperModel
from multiprocessing import Queue

def transcription_worker(audio_queue: Queue, text_queue: Queue):
    # Load Whisper on GPU
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Whisper will run on CPU.")

    model = WhisperModel("large-v2", device="cuda", compute_type="float16")
    print("Whisper model loaded on GPU.")

    while True:
        audio = audio_queue.get()  # blocking
        segments, _ = model.transcribe(audio, language="en", beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        if text:
            print("\n--- TRANSCRIPTION ---\n", text)
            text_queue.put(text)
