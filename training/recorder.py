import sounddevice as sd
import scipy.io.wavfile as wav

sample_rate = 16000

input("Press enter, then say Hey Su Lei...")

audio = sd.rec(
    int(1 * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="int16"
)

sd.wait()

wav.write(
    "hey_su_lei.wav",
    sample_rate,
    audio
)