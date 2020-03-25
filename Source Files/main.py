# Imports needed for NLP and TensorFlow
import json
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Import the chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

try:
    with open("training_data.pickle", "rb") as f:
        data = pickle.load(f)
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']

except:
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add the word to the words list
            words.extend(w)
            # add the word to documents in our corpus
            documents.append((w, intent['tag']))
            # add the tag to the class list if it does not exist yet
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    # remove duplicate words
    words = sorted(list(set(words)))

    # remove duplicate classes
    classes = sorted(list(set(classes)))

    '''print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique stempped words", words)'''

    # Tranform data further from documents of words into tensors of numbers.
    # Create the training data
    training = []
    output = []

    # create an empty array for the output
    output_empty = [0 for _ in range(len(classes))]

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize the bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # steam each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create the bag of words array
        # 1 is added to the bag of words if the word exists else 0
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle the features and turn into np.array
    # random.shuffle(training)
    training = np.array(training)
    output = np.array(output)

    # create train and test lists
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Save the data structures
    with open("training_data.pickle", "wb") as f:
        pickle.dump({'words': words, 'classes': classes,
                     'train_x': train_x, 'train_y': train_y}, f)

# reset underlying graph data
tf.reset_default_graph()

# Define the input shape to be expected by the model
# Input layer
net = tflearn.input_data(shape=[None, len(train_x[0])])
# Fully connected hidden layers with 8 neurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# Output layer with softmax activation
net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Training
try:
    model.load("chatbot.tflearn")
except:
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("chatbot.tflearn")


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words, show_details=False):
    bag = [0 for _ in range(len(words))]

    sentence_words = clean_up_sentence(sentence)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def chat():
    print("Start talking with ChatBot!")
    print("Type quit to stop chatting.")

    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = classes[results_index]

        if results[results_index] > 0.7:
            for tg in intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I'm sorry, I didn't get that. Try again.")


chat()
