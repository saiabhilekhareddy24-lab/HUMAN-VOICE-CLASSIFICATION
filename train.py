import os
import numpy as np
import joblib

from features import extract_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset"

classes = {
    "male": 0,
    "female": 1
}

X = []
y = []

print("=" * 45)
print(" HUMAN VOICE CLASSIFICATION")
print("=" * 45)

print("\nLoading dataset...")

for class_name, label in classes.items():

    folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(folder):
        print(f"Warning: {folder} not found")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(f"{class_name.capitalize()} samples : {len(files)}")

    for file in files:
        file_path = os.path.join(folder, file)

        try:
            feature = extract_features(file_path)
            X.append(feature)
            y.append(label)

        except Exception as error:
            print(f"Error processing {file}: {error}")


X = np.array(X)
y = np.array(y)

if len(X) < 4:
    raise RuntimeError(
        "Not enough audio samples. Add more .wav files."
    )

print("\nExtracting audio features...")
print("Feature extraction completed.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

train_accuracy = accuracy_score(
    y_train,
    train_prediction
)

test_accuracy = accuracy_score(
    y_test,
    test_prediction
)

print("\nModel Training Completed.")
print(f"\nTraining Accuracy : {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_prediction,
        target_names=["Male", "Female"]
    )
)

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/voice_classifier.pkl"
)

print("\nModel saved to:")
print("model/voice_classifier.pkl")
