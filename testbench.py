import os
import joblib
import numpy as np

from features import extract_features


MODEL_PATH = "model/voice_classifier.pkl"

CLASS_NAMES = {
    0: "Male",
    1: "Female"
}


def run_test(audio_file, expected_class=None):

    print("\n----------------------------------------")
    print(f"Test File: {audio_file}")

    if not os.path.exists(audio_file):
        print("Result: SKIPPED")
        print("Reason: Audio file not found")
        return

    try:
        model = joblib.load(MODEL_PATH)

        features = extract_features(audio_file)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        predicted_class = CLASS_NAMES[prediction]
        confidence = probabilities[prediction] * 100

        print(f"Expected  : {expected_class}")
        print(f"Predicted : {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

        if expected_class is None:
            print("Result    : PASS")
        elif predicted_class.lower() == expected_class.lower():
            print("Result    : PASS")
        else:
            print("Result    : FAIL")

    except Exception as error:
        print(f"Result: FAIL")
        print(f"Error: {error}")


print("=" * 40)
print(" HUMAN VOICE CLASSIFICATION TESTBENCH")
print("=" * 40)

if not os.path.exists(MODEL_PATH):

    print("\nERROR:")
    print("Trained model does not exist.")
    print("Run 'python train.py' first.")

else:

    run_test(
        "dataset/male/test.wav",
        "Male"
    )

    run_test(
        "dataset/female/test.wav",
        "Female"
    )

print("\n========================================")
print(" TESTBENCH COMPLETED")
print("========================================")
