import torch
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm

from openwakeword.utils import AudioFeatures


# -----------------------------
# Define your trained network
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


# -----------------------------
# Load trained weights
# -----------------------------

checkpoint = torch.load(
    "models/wakeword_model.pth",
    map_location="cpu"
)

fcn.load_state_dict(checkpoint)
fcn.eval()

print("Wakeword model loaded!")


# -----------------------------
# Load OpenWakeWord feature extractor
# -----------------------------

F = AudioFeatures()

print("Audio feature extractor loaded!")

def predict_audio(path):

    # Load audio exactly like training
    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    print("Original length:", len(audio)/16000)


    # convert to int16
    audio = (audio * 32767).astype(np.int16)


    # create 3 second window
    target_len = 16000 * 3


    # If audio is shorter than 3 sec, pad at the beginning
    if len(audio) < target_len:

        padded = np.zeros(
            target_len,
            dtype=np.int16
        )

        # put audio at END (same as training)
        padded[-len(audio):] = audio

        audio = padded


    # If audio is longer than 3 sec, take the last 3 sec
    elif len(audio) > target_len:

        audio = audio[-target_len:]


    print(
        "After padding:",
        len(audio)/16000
    )


    # embeddings
    features = F._get_embeddings(audio)

    print(
        "Embedding shape:",
        features.shape
    )


    scores = []


    # Sliding window prediction
    for i in tqdm(range(features.shape[0]-27)):

        window = features[i:i+28][None,:,:]


        with torch.no_grad():

            score = fcn(
                torch.from_numpy(window).float()
            )


        scores.append(score.item())


    print("Scores:", scores)

    if len(scores) > 0:
        print("Max confidence:", max(scores))
    else:
        print("No prediction windows")


# Test noise
predict_audio("./training/negative/noise.wav")