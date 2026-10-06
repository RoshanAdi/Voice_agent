import queue
import time
import yaml
import numpy as np
import sounddevice as sd

from dsp import NoiseSuppressor


class MicNoiseSuppressor:
    """
    Captures microphone audio, applies noise suppression,
    and exposes filtered audio frames via a queue.
    """

    def __init__(self, config_path="config.yaml"):
        self.cfg = self._load_config(config_path)

        self.sr = int(self.cfg["samplerate"])
        self.frame_ms = int(self.cfg["frame_ms"])

        self.ns = NoiseSuppressor(
            sr=self.sr,
            frame_ms=self.frame_ms,
            beta=self.cfg["noise_beta"],
            noise_floor=self.cfg["noise_floor"],
            ema_alpha=self.cfg["ema_alpha"],
            gain_smooth=self.cfg["gain_smooth"],
            highpass_hz=self.cfg["highpass_hz"],
        )

        self.hop = self.ns.hop
        self.audio_q = queue.Queue(maxsize=32)
        self.stream = None

    def _load_config(self, path):
        cfg = {
            "samplerate": 16000,
            "frame_ms": 20,
            "calib_sec": 1.0,
            "highpass_hz": 150,
            "noise_beta": 1.5,
            "noise_floor": 0.01,
            "ema_alpha": 0.95,
            "gain_smooth": 0.7,
            "device_in": "default",
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                cfg.update(loaded)
        except FileNotFoundError:
            pass
        return cfg

    def _callback(self, indata, frames, time_info, status):
        if status:
            return

        x = indata[:, 0].copy()

        if len(x) != self.hop:
            x = x[:self.hop] if len(x) > self.hop else np.pad(x, (0, self.hop - len(x)))

        GAIN = 22.0  # adjust as needed

        y = self.ns.process(x)
        y = np.clip(y * GAIN, -1.0, 1.0)  # avoid clipping

        try:
            self.audio_q.put_nowait(y)
        except queue.Full:
            pass

    def calibrate(self):
        """
        Noise calibration phase (silence required)
        """
        t_end = time.time() + float(self.cfg["calib_sec"])
        while time.time() < t_end:
            buf = sd.rec(
                self.hop,
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                device=self.cfg["device_in"],
            )
            sd.wait()
            self.ns.calibrate_noise(buf[:, 0])

    def start(self):
        """
        Start microphone capture + noise suppression
        """
        self.calibrate()
        self.data = None
        self.stream = sd.InputStream(
            device=self.cfg["device_in"],
            samplerate=self.sr,
            blocksize=self.hop,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_frame(self, timeout=None):
        """
        Get next filtered audio frame (NumPy float32 array)
        Returns None if timeout
        """
        try:
            return self.audio_q.get(timeout=timeout)
        except queue.Empty:
            return None
