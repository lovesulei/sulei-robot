import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path

SAMPLE_RATE = 48000
DURATION = 1.0  # seconds

output_dir = Path("training/positive")
output_dir.mkdir(parents=True, exist_ok=True)

# Find next available number
i = 1
while (output_dir / f"su_lei_{i:03d}.wav").exists():
    i += 1

print("Recording dataset for wake word: 'Su Lei'")
print("Press Enter to record.")
print("Type 'q' then Enter to quit.\n")

while True:
    cmd = input(f"[{i:03d}] Ready? ")

    if cmd.lower() == "q":
        break

    print("Speak now...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    filename = output_dir / f"su_lei_{i:03d}.wav"

    wav.write(
        filename,
        SAMPLE_RATE,
        audio,
    )

    print(f"✓ Saved {filename}\n")

    i += 1

print("Done!")