import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import random

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

# Use the learning rate schedule in the optimizer
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

# Initialize the lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load data
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Data preprocessing
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Initialize training data
X = []
Y = []
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    X.append(bag)
    Y.append(output_row)

# Shuffle and convert to NumPy arrays
temp = list(zip(X, Y))
random.shuffle(temp)
X, Y = zip(*temp)

train_x = np.array(X)
train_y = np.array(Y)

# Model definition
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model

model.save('chatbot_model.h5')

print("Model created")
