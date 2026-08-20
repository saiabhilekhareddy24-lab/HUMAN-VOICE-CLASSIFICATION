import sys
import os
import joblib

from features import extract_features


MODEL_PATH = "model/voice_classifier.pkl"

class_names = {
    0: "Male",
    1: "Female"
}


def predict_voice(audio_file):

    if not os.path.exists(audio_file):
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    features = extract_features(audio_file)

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = probabilities[prediction] * 100

    return class_names[prediction], confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python predict.py test_voice.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    predicted_class, confidence = predict_voice(
        audio_file
    )

    print("\n========================================")
    print(" VOICE CLASSIFICATION RESULT")
    print("========================================")

    print(f"\nAudio File : {audio_file}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")
