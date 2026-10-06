import numpy as np
import torch
from faster_whisper import WhisperModel
import threading
import queue
from mic_noise_suppressor import MicNoiseSuppressor
import time
from llm_engine import LLMAssistant
import sounddevice as sd
from piper.voice import PiperVoice
from silero_vad import load_silero_vad, get_speech_timestamps

FRAME_MS = 20  # responsiveness
stop_tts_event = threading.Event()
Update_llm_user_disturbed = threading.Event()
accept_voice = threading.Event()
tts_in_progress = threading.Event()
SAMPLE_RATE = 16000
SILENCE_SECONDS = 2
SILENCE_SAMPLES = SAMPLE_RATE * SILENCE_SECONDS
SILENCE_THRESHOLD = 0.02  # energy threshold for silence

audio_queue = queue.Queue()
text_queue = queue.Queue()
tts_queue = queue.Queue()

def is_silent(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms < SILENCE_THRESHOLD

def has_voice(audio):
    rms = np.sqrt(np.mean(audio ** 2))
    return rms >= SILENCE_THRESHOLD

def clear_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break


def audio_capture_thread(ns):
    buffer = []
    buffer_samples = 0
    silence_samples = 0
    not_silent_counter = 0
    model = load_silero_vad()
    while True:
        frame = ns.get_frame(timeout=1.0)
        if frame is None:
            continue

        buffer.append(frame)
        buffer_samples += len(frame)

        if is_silent(frame):
            silence_samples += len(frame)
            not_silent_counter = 0
        else:
            silence_samples = 0
            not_silent_counter += 1

            # Only set stop_tts_event if we detect actual human speech
        if not_silent_counter >= 10:
            audio = np.concatenate(buffer)
            speech_timestamps = get_speech_timestamps(
                audio,
                model,
                return_seconds=True
            )
            if len(speech_timestamps) > 0:
                stop_tts_event.set()
                accept_voice.set()
            else:
                # No human speech, don't trigger
                stop_tts_event.clear()
        

        # If 2 seconds of silence detected → finalize chunk
        if silence_samples >= SILENCE_SAMPLES and buffer_samples > 0 and accept_voice.is_set():
            audio = np.concatenate(buffer)
            stop_tts_event.clear()
            accept_voice.clear()


            if has_voice(audio):
                speech_timestamps = get_speech_timestamps(
                    audio,
                    model,
                    return_seconds=True,
                )
                print(f"Detected speech timestamps: {speech_timestamps}")
                if len(speech_timestamps) > 0:
                    audio_queue.put(audio)

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
        if text .strip() != "":
            text_queue.put(text)
        audio_queue.task_done()

def llm_thread(text_queue, llm):
    while True:
        text = text_queue.get()

        system_instruction = None
        if Update_llm_user_disturbed.is_set():
            system_instruction = (
                "The user started speaking before your previous reply was delivered. "
                "Assume the user did NOT hear your last response. "
                "Generate the next reply without relying on it."
            )
            Update_llm_user_disturbed.clear()

        output = llm.get_reply(
            user_input=text,
            system_instruction=system_instruction
        )

        tts_queue.put(output)
        print(f"Assistant: {output}")
        text_queue.task_done()



def tts_thread(tts_queue):
    voice = PiperVoice.load(
        "en_US-amy-medium.onnx",
        "en_US-amy-medium.onnx.json"
    )

    while True:
        text = tts_queue.get()
        stop_tts_event.clear()
        tts_in_progress.set()
        sr = voice.config.sample_rate
        frame_size = int(sr * FRAME_MS / 1000)

        with sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            blocksize=frame_size  # important
        ) as stream:

            for chunk in voice.synthesize(text):

                audio = np.array(chunk.audio_float_array, dtype=np.float32)

                # split chunk into small frames
                for i in range(0, len(audio), frame_size):

                    if stop_tts_event.is_set():
                        print("TTS interrupted")
                        Update_llm_user_disturbed.set()
                        clear_queue(tts_queue)
                        break

                    frame = audio[i:i+frame_size]
                    stream.write(frame)

                if stop_tts_event.is_set():
                    break

        tts_queue.task_done()
        tts_in_progress.clear()







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
    reference_path = "recording.m4a"

    assistant = LLMAssistant()
    
    print("Mic noise suppression started... Speak into the mic.")

    # Start threads
    t1 = threading.Thread(target=audio_capture_thread, args=(ns,), daemon=True)
    t2 = threading.Thread(target=transcription_thread, args=(model,), daemon=True)
    t3 = threading.Thread(target=llm_thread,args=(text_queue, assistant),daemon=True)
    t4 = threading.Thread(target=tts_thread,args=(tts_queue,),daemon=True)

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping…")
        ns.stop()


if __name__ == "__main__":
    main()
