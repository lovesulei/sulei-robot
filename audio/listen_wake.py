import sounddevice as sd
import numpy as np
from openwakeword.model import Model
from scipy.signal import resample_poly

MIC = 1
INPUT_RATE = 48000
WAKE_RATE = 16000

model = Model()

print("Listening for wake word...")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # convert 48kHz -> 16kHz
    audio = indata[:, 0]

    audio = resample_poly(
        audio,
        WAKE_RATE,
        INPUT_RATE
    )

    audio = (audio * 32767).astype(np.int16)
    print("audio samples:", len(audio))
    prediction = model.predict(audio)

    for word, score in prediction.items():
        if score > 0.5:
            print("Wake detected:", word, score)


with sd.InputStream(
    device=MIC,
    channels=1,
    samplerate=INPUT_RATE,
    blocksize=3840,
    dtype="float32",
    callback=audio_callback,
):
    while True:
        sd.sleep(1000)