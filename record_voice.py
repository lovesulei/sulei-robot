import sounddevice as sd
import soundfile as sf


RATE = 16000


def record(filename, seconds=3):
    print(f"Recording {filename}")
    print("Speak now!")

    audio = sd.rec(
        int(seconds * RATE),
        samplerate=RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    sf.write(
        filename,
        audio,
        RATE
    )

    print("Saved!")


if __name__ == "__main__":

    record(
        "audio/greetings/hi_sulei.wav",
        2
    )