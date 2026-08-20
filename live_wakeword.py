import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly
import torch
import torch.nn as nn
from collections import deque
from openwakeword.utils import AudioFeatures


# -----------------------------
# Load trained model
# -----------------------------

layer_dim = 32

fcn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*96, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, layer_dim),
    nn.LayerNorm(layer_dim),
    nn.ReLU(),
    nn.Linear(layer_dim, 1),
    nn.Sigmoid(),
)

checkpoint = torch.load(
    "models/wakeword_model.pth",
    map_location="cpu"
)

fcn.load_state_dict(checkpoint)
fcn.eval()

print("Wakeword model loaded!")


# -----------------------------
# Feature extractor
# -----------------------------

F = AudioFeatures()

print("Feature extractor loaded!")


# -----------------------------
# Audio settings
# -----------------------------

DEVICE = 1

INPUT_RATE = 44100
TARGET_RATE = 16000

WINDOW_SECONDS = 3

BUFFER_SIZE = TARGET_RATE * WINDOW_SECONDS

THRESHOLD = 0.8


# Store last 3 seconds
audio_buffer = deque(
    maxlen=BUFFER_SIZE
)


# -----------------------------
# Microphone callback
# -----------------------------

def callback(indata, frames, time, status):

    if status:
        print(status)

    # float32 from mic
    audio = indata[:,0]


    # 44.1k -> 16k
    audio_16k = resample_poly(
        audio,
        TARGET_RATE,
        INPUT_RATE
    )


    audio_16k = (
        audio_16k * 32767
    ).astype(np.int16)


    for sample in audio_16k:
        audio_buffer.append(sample)



# -----------------------------
# Prediction
# -----------------------------

def predict():

    # wait until we have a full 3 seconds
    if len(audio_buffer) < BUFFER_SIZE:
        return 0

    audio = np.array(audio_buffer)

    features = F._get_embeddings(audio)

    if features.shape[0] < 28:
        return 0

    window = features[-28:][None,:,:]

    with torch.no_grad():
        score = fcn(
            torch.from_numpy(window).float()
        )

    return score.item()



# -----------------------------
# Start listening
# -----------------------------

print("Listening... Say your wake word!")

with sd.InputStream(
    device=DEVICE,
    samplerate=INPUT_RATE,
    channels=1,
    dtype="float32",
    callback=callback
):

    while True:

        score = predict()

        print(
            f"Confidence: {score:.3f}"
        )


        if score > THRESHOLD:

            print("WAKE WORD DETECTED!")

            # TODO:
            # play response
            # move servo
            # activate animation


        sd.sleep(500)