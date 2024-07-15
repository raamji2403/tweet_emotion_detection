Emotion Detection with LSTM Model
Overview
This project aims to build an emotion detection model using LSTM (Long Short-Term Memory) networks. The model is trained on a dataset of tweets labeled with different emotions. It is capable of predicting the emotion of a given tweet or sentence.

Requirements
Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
Project Structure
train.txt, val.txt, test.txt: These files contain the training, validation, and test datasets, respectively. Each line in these files has the format tweet;label.
emotion_detection.py: The main script for training the model and predicting emotions.
emotion_detection.ipynb: Jupyter notebook with the same functionality as the main script for easier experimentation and visualization.
Setup
Clone the repository.
Install the required packages:
bash
Copy code
pip install tensorflow numpy matplotlib
Place your dataset files (train.txt, val.txt, test.txt) in the project directory.
Usage
Training the Model
To train the model, run the emotion_detection.py script:

bash
Copy code
python emotion_detection.py
This script will:

Load the training, validation, and test datasets.
Preprocess the data by tokenizing the tweets and converting labels to indices.
Build and compile an LSTM model.
Train the model on the training data and validate it on the validation data.
Predicting Emotions
After training the model, you can use the same script to predict emotions from the command line:

bash
Copy code
python emotion_detection.py
Enter a sentence when prompted, and the model will predict the emotion. Type 0 to quit.

Using the Jupyter Notebook
For experimentation and visualization, you can use the Jupyter notebook:

bash
Copy code
jupyter notebook emotion_detection.ipynb
The notebook contains the same steps as the script, with additional visualizations and explanations.

Data Preprocessing
Loading Data: The datasets are loaded from text files.
Splitting Data: Tweets and labels are extracted from each line in the datasets.
Tokenizing Tweets: Tweets are tokenized and converted to sequences of integers.
Padding Sequences: Sequences are padded to a fixed length.
Mapping Labels: Labels are mapped to numerical indices.
Model Architecture
The model consists of:

An Embedding layer with an input dimension of 10,000 and output dimension of 16.
Two Bidirectional LSTM layers with 20 units each.
A Dense layer with a softmax activation function for multi-class classification.
Training
The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. The training process includes validation on a separate validation dataset to monitor performance and avoid overfitting.

Evaluation
The model's performance can be evaluated using the test dataset. After training, you can visualize the training and validation accuracy and loss over epochs.

Contribution
Feel free to fork the repository and contribute by submitting pull requests.
