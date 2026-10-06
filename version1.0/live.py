import numpy as np
import soundfile as sf
import time

from mic_noise_suppressor import MicNoiseSuppressor

SAMPLE_RATE = 16000
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

ns = MicNoiseSuppressor()
ns.start()

buffer = []
buffer_samples = 0
clip_index = 0

try:
    while True:
        frame = ns.get_frame(timeout=1.0)
        if frame is None:
            continue

        buffer.append(frame)
        buffer_samples += len(frame)

        if buffer_samples >= CLIP_SAMPLES:
            audio = np.concatenate(buffer)[:CLIP_SAMPLES]

            filename = f"clean_clip_{clip_index:04d}.wav"
            sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")

            print(f"Saved {filename}")

            clip_index += 1
            buffer = []
            buffer_samples = 0

except KeyboardInterrupt:
    print("Stopping…")
    ns.stop()
