Human Voice Classification
Overview

Human Voice Classification is a machine learning project that classifies human speech using audio signal processing and machine learning techniques. The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features from voice recordings and uses a Random Forest classifier to identify the voice class.

Features
Audio file preprocessing
MFCC feature extraction
Machine learning-based voice classification
Training and testing support
Accuracy calculation
Confusion matrix generation
HTML-based simulation results
Technologies Used
Python
Librosa
NumPy
Scikit-learn
Matplotlib
Joblib
HTML/CSS
Project Workflow
Voice Input
     ↓
Audio Preprocessing
     ↓
MFCC Feature Extraction
     ↓
Feature Vector
     ↓
Random Forest Classifier
     ↓
Predicted Voice Class
     ↓
Simulation Output

Dataset

The dataset should contain WAV audio files arranged into folders according to their classes.

Example:

dataset/
├── male/
│   ├── voice1.wav
│   ├── voice2.wav
│   └── voice3.wav
│
└── female/
    ├── voice1.wav
    ├── voice2.wav
    └── voice3.wav


The folder names are automatically treated as classification labels.

Installation

Install the required Python libraries:

pip install -r requirements.txt

Training the Model

Run:

python train.py


The trained model will be saved in:

models/voice_classifier.pkl

Voice Prediction

To classify a new audio file:

python predict.py sample_audio/sample.wav


Example output:

Predicted Class: male
Confidence: 94.27%

Testing

Run the testbench:

python testbench.py


The testbench evaluates the trained classifier using the test dataset and generates classification results.

Simulation Output

The HTML simulation report is generated at:

results/simulation_output.html


It displays:

Number of test samples
Correct predictions
Incorrect predictions
Classification accuracy
Individual prediction results
Applications

Human voice classification can be used in:

Speaker identification
Voice-based authentication
Human-computer interaction
Smart assistants
Audio analysis
Security systems
Speech processing applications
Future Improvements
Deep learning using CNN/LSTM models
Real-time microphone input
Speaker identification
Noise reduction
Web-based voice classification
Mobile application integration
License

This project is intended for educational and research purposes.
