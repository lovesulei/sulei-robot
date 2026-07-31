import sounddevice as sd
import numpy as np
import wave

duration = 5
sample_rate = 48000

print("Recording...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype=np.int16,
    device=1
)

sd.wait()

print("Saving...")

with wave.open("test.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(audio.tobytes())

print("Done!")