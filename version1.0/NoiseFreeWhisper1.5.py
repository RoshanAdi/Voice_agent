from urllib import response
import numpy as np
import torch
from faster_whisper import WhisperModel
import threading
import queue
from mic_noise_suppressor import MicNoiseSuppressor
import time
from llama_cpp import Llama

SAMPLE_RATE = 16000
SILENCE_SECONDS = 2
SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_SECONDS
SILENCE_THRESHOLD = 0.02  # energy threshold for silence

audio_queue = queue.Queue()
text_queue = queue.Queue()

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

        segments, info = model.transcribe(audio, language="en", beam_size=5)
        text = " ".join([segment.text for segment in segments])
        print("\n--- TRANSCRIPTION ---")
        print(text)
        text_queue.put(text)
        audio_queue.task_done()

def llm_thread(text_queue, llm):

    while True:
        text = text_queue.get()

        prompt  = text
        output = llm(
            prompt,
            max_tokens=128,
            temperature=0.5,
            top_p=0.5,
            repeat_penalty=1,
            stop=["Question:", "\n\n"],
            echo=False)

        print(output["choices"][0]["text"].strip())

        text_queue.task_done()


def main():
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Whisper will run on CPU.")

    print("Using GPU:", torch.cuda.get_device_name(0))

    # Load Whisper medium model on GPU
    model = WhisperModel("large-v2", device="cuda",device_index=1, compute_type="float16")  

    # Start noise suppression mic pipeline
    ns = MicNoiseSuppressor()
    ns.start()

    llm = Llama(
    model_path="./llama-7b.Q4_K_M.gguf",
    n_gpu_layers=-1,main_gpu=0 )
    
    print("Mic noise suppression started... Speak into the mic.")

    # Start threads
    t1 = threading.Thread(target=audio_capture_thread, args=(ns,), daemon=True)
    t2 = threading.Thread(target=transcription_thread, args=(model,), daemon=True)
    t3 = threading.Thread(target=llm_thread,args=(text_queue, llm),daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping…")
        ns.stop()


if __name__ == "__main__":
    main()
