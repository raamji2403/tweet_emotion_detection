# Emotion Detection with LSTM Model

This project implements an emotion detection model using LSTM (Long Short-Term Memory) networks with TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset Requirements](#dataset-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
  
## Introduction
The Emotion Detection with LSTM Model project aims to classify emotions from textual data using deep learning techniques. It utilizes bidirectional LSTM layers for sequence processing and softmax activation for multi-class classification.

## Features
- Tokenization and sequence padding of textual data
- Bidirectional LSTM architecture for capturing sequence information
- Training, validation, and testing of the emotion detection model
- Command-line interface for predicting emotions from user input sentences

## Dataset Requirements
- The project expects training, validation, and test datasets in the format:
  - `train.txt`, `val.txt`, `test.txt`: Each line in these files should contain a tweet followed by a label separated by ';'.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/emotion-detection-lstm.git
   cd emotion-detection-lstm
2. Install the required Python packages:
   ```bash
   pip install tensorflow numpy matplotlib 
## Usage
Training the Model
Prepare your dataset files (train.txt, val.txt, test.txt) in the project directory.
Run the training script:
bash
Copy code
python emotion_detection.py
This script will load, tokenize, pad sequences, build, compile, and train the LSTM model.
Training progress and validation accuracy will be displayed.
Predicting Emotions
After training, you can predict emotions using the trained model:
bash
Copy code
python emotion_detection.py
Enter a sentence when prompted. The model will predict the emotion based on the input.
## Repository Structure
tweet_emotion_detection/
├── tweet_emotion_detection.py
├── tweet_emotion_detection.ipynb
├── train.txt
├── val.txt
├── test.txt
└── README.md
emotion_detection.py: Main script for model training and prediction.
train.txt, val.txt, test.txt: Dataset files containing tweets and labels.
README.md: This file providing project overview, installation, and usage instructions.
## Contributing
Contributions are welcome! Please feel free to open issues or pull requests for any improvements or features.
