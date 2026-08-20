import librosa
import numpy as np


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )
    mfcc_features = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )
    chroma_features = np.mean(chroma.T, axis=0)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate
    )
    mel_features = np.mean(mel.T, axis=0)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate
    )
    contrast_features = np.mean(contrast.T, axis=0)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_feature = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )
    centroid_feature = np.mean(centroid)

    features = np.hstack([
        mfcc_features,
        chroma_features,
        mel_features,
        contrast_features,
        zcr_feature,
        centroid_feature
    ])

    return features
