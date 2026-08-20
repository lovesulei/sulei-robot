import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
from scipy.io.wavfile import write

MIC_DEVICE = 1
MIC_RATE = 44100
WHISPER_RATE = 16000


def record_audio(filename="command.wav", duration=5):
    print("Listening...")

    audio = sd.rec(
        int(duration * MIC_RATE),
        samplerate=MIC_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
    )

    sd.wait()

    # 44.1 kHz -> 16 kHz
    audio_16k = resample_poly(
        audio[:, 0],
        WHISPER_RATE,
        MIC_RATE,
    )

    audio_16k = (
        audio_16k * 32767
    ).astype(np.int16)

    write(filename, WHISPER_RATE, audio_16k)

    print("Recording finished.")

    return filename