from contextlib import nullcontext
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers import add
from keras.utils import to_categorical, plot_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
import cv2
import string
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Model_with_attention_improved_2 import ReduceSumLayer
import pickle

def clean_description(desc):
    # Clean descriptions by removing punctuation and non-alphabetic words
    for key, des in desc.items():
        if isinstance(des, str):  # Ensure the value is a string
            caption = des.translate(str.maketrans('', '', string.punctuation))
            caption = caption.split()  # Split into words
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            desc[key] = " ".join(caption)  # Rejoin words into a string

def to_vocab(desc):
    # Create a set of unique words from descriptions
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    return words

def load_clean_descriptions(des, dataset):
    # Create dictionary of cleaned descriptions for dataset images
    dataset_des = dict()
    dataset = set(dataset)  # Keep image names unchanged (with .jpg)
    print("Dataset keys (unchanged):", list(dataset)[:5])  # Debug print
    for key, des_list in des.items():
        if key in dataset:
            if key not in dataset_des:
                dataset_des[key] = list()
            for line in des_list:
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc)
    return dataset_des

def preprocess_img(img_path):
    # Load and preprocess image to match InceptionV3 input requirements (299x299x3)
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)
    return x

def encode(image, model):
    # Extract features from an image using the provided model
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def greedy_search(pic, base_model, model, wordtoix, ixtoword, max_length):
    # Generate caption for an image using greedy search
    # Check if input is a file path or image data
    if isinstance(pic, str):
        img = cv2.imread(pic)  # Load image from file path
        if img is None:
            print(f"Error: Could not read image from path: {pic}")
            return None
    else:
        image_stream = pic.read()
        print(f"Size of image file data: {len(image_stream)} bytes")  # Debug: Check data size
        np_arr = np.frombuffer(image_stream, np.uint8)
        if np_arr.size == 0:
            print("Error: Image buffer is empty!")
            return None
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Failed to decode image!")
            return None

    # Resize and preprocess image
    img = cv2.resize(img, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    # Extract image features
    features = base_model.predict(img)
    features = features.reshape((1, 2048))  # Reshape for model input

    # Generate caption
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([features, seq])  # Predict next word
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break

    final = start.split()
    final = final[1:-1]  # Remove 'startseq' and 'endseq'
    final = ' '.join(final)
    return final

base_model = None
model = None

def getDescription(image_file):
    global base_model, model
    # Load base model if not already loaded
    if base_model is None:
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    # Load word-to-index and index-to-word mappings
    wordtoix_path = './wordtoix.pkl'
    ixtoword_path = './ixtoword.pkl'
    max_length_path = './max_length.txt'

    # Load main model with custom layer
    if model is None:
        model = load_model('best_model_attention.keras', custom_objects={"ReduceSumLayer": ReduceSumLayer})

    with open(wordtoix_path, 'rb') as f:
        wordtoix = pickle.load(f)
    with open(ixtoword_path, 'rb') as f:
        ixtoword = pickle.load(f)
    with open(max_length_path, 'r') as f:
        max_length = int(f.read())

    # Generate and return caption for the image
    return greedy_search(image_file, base_model, model, wordtoix, ixtoword, max_length)