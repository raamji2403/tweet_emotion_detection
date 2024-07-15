import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data from a file
def load_data(file_path):
    with open(file_path, "r") as file:
        return file.readlines()
    
# Load datasets
train_file = "./train.txt"
val_file = "./val.txt"
test_file = "./test.txt"
train = load_data(train_file)
val = load_data(val_file)
test = load_data(test_file)

# Split data into tweets and labels
def get_tweet(data):
    tweets = []
    labels = []
    for line in data:
        tweet, label = line.strip().split(';')
        tweets.append(tweet)
        labels.append(label)
    return tweets, labels


# Extract tweets and labels
tweets, labels = get_tweet(train)
val_tweets, val_labels = get_tweet(val)
test_tweets, test_labels = get_tweet(test)

# Tokenize tweets
def get_sequences(tokenizer, tweets, maxlen=50):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=maxlen)
    return padded

# Tokenize tweets
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(tweets)
padded_train_seq = get_sequences(tokenizer, tweets)
padded_val_seq = get_sequences(tokenizer, val_tweets)
padded_test_seq = get_sequences(tokenizer, test_tweets)

# Create label index mappings
classes = set(labels)
class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {i: c for c, i in class_to_index.items()}
names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])

# Convert labels to indices
train_labels = names_to_ids(labels)
val_labels = names_to_ids(val_labels)
test_labels = names_to_ids(test_labels)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
h = model.fit(padded_train_seq, train_labels, validation_data=(padded_val_seq, val_labels), epochs=20)

# Function to predict emotion from the command line
def predict_emotion():
    while True:
        sentence = input("Enter a sentence (0 to quit): ")
        if sentence == "0":
            break
        if sentence == "":
            print("Enter the Sentence..!!")
            continue
        seq = tokenizer.texts_to_sequences([sentence])
        seq = pad_sequences(seq, maxlen=50, padding='post')
        p = model.predict(seq)[0]
        pred_class = index_to_class[np.argmax(p).astype('uint8')].capitalize()
        print(f"Predicted emotion: {pred_class}")

# Start the prediction loop
predict_emotion()
