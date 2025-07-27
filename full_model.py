from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, RepeatVector, TimeDistributed, Multiply, Add, Softmax, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.src.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import string
import pandas as pd
import os
import random

# Initialize random seeds for reproducibility across multiple runs
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Custom ReduceSumLayer for attention mechanism
@register_keras_serializable(package="Custom")
class ReduceSumLayer(Layer):
    def __init__(self, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Sum the inputs along axis 1
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        config = super(ReduceSumLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        from keras.mixed_precision import Policy
        if 'dtype' in config and isinstance(config['dtype'], dict):
            config['dtype'] = Policy(config['dtype']['config']['name'])
        return cls(**config)

# Custom data generator for batch processing
class DataGenerator(Sequence):
    def __init__(self, image_features, descriptions, wordtoix, max_length, batch_size):
        self.image_features = image_features
        self.descriptions = list(descriptions.items())
        self.wordtoix = wordtoix
        self.max_length = max_length
        self.batch_size = batch_size
        self.data = self._create_data_index()

    def _create_data_index(self):
        # Create list of (image, input sequence, output word) tuples
        data = []
        for img, caps in self.descriptions:
            for cap in caps:
                seq = [self.wordtoix[word] for word in cap.split() if word in self.wordtoix]
                for i in range(1, len(seq)):
                    data.append((img, seq[:i], seq[i]))
        return data

    def __len__(self):
        # Return number of batches
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # Generate a batch of data
        batch = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]
        X1, X2, y = [], [], []
        for img, in_seq, out_word in batch:
            X1.append(self.image_features[img])
            X2.append(tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=self.max_length)[0])
            y.append(out_word)
        return ((np.array(X1, dtype=np.float32),
                 np.array(X2, dtype=np.int32)),
                np.array(y, dtype=np.int32))

# Global configuration
BATCH_SIZE = 64
EPOCHS = 20
EMBEDDING_DIM = 200
FEATURE_SIZE = 2048

# Load data from Excel file
data = pd.read_excel('./data.xlsx')

def load_description(data):
    # Create a mapping of image IDs to their descriptions
    mapping = dict()
    for line in range(1, 158914):
        try:
            token = data[line].split("|")
            img_id = token[0]
            des_num = token[1]
            imd_des = token[2]
            if img_id not in mapping:
                mapping[img_id] = list()
                mapping[img_id].append(imd_des)
            else:
                mapping[img_id].append(imd_des)
        except IndexError:
            print(f"Skipping line {line} due to IndexError")
            continue
    return mapping

# Extract descriptions from Excel column
descriptions = load_description(data["Column1"])

def clean_description(desc):
    # Clean descriptions by removing punctuation and non-alphabetic words
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string.punctuation]
            caption = ''.join(caption)
            caption = caption.split(' ')
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption

# Clean the descriptions
clean_description(descriptions)

def to_vocab(desc):
    # Create a set of unique words from descriptions
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    return words

# Generate vocabulary from descriptions
vocab = to_vocab(descriptions)

# Select 10,000 images for training
images = os.listdir('./flickr30k_images')
train_img = []
for i in range(10000):
    train_img.append(images[i])

def load_clean_descriptions(des, dataset):
    # Create dictionary of cleaned descriptions for training images
    dataset_des = dict()
    dataset = set(dataset)
    print("Dataset keys (unchanged):", list(dataset)[:5])
    for key, des_list in des.items():
        if key in dataset:
            if key not in dataset_des:
                dataset_des[key] = list()
            for line in des_list:
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc)
    return dataset_des

# Create dictionary of image names and descriptions for training
train_descriptions = load_clean_descriptions(descriptions, train_img)

# Load InceptionV3 model for feature extraction
base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
model = Model(base_model.input, base_model.layers[-2].output)

def preprocess_img(img_path):
    # Load and preprocess image to match InceptionV3 input requirements
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image, model):
    # Extract features from an image
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

# Extract features for all training images
encoding_train = {}
image_dir = "./flickr30k_images"
for img in train_img:
    image_path = os.path.join(image_dir, img)
    encoding_train[img[len(images):]] = encode(image_path, base_model)

# Create list of all training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for caption in val:
        all_train_captions.append(caption)

# Count word frequencies in captions
vocabulary = vocab
threshold = 7
word_counts = {}
for cap in all_train_captions:
    for word in cap.split(' '):
        word_counts[word] = word_counts.get(word, 0) + 1

# Filter words with frequency less than 7
vocab = [word for word in word_counts if word_counts[word] >= threshold]

# Create word-to-index and index-to-word dictionaries
ixtoword = {}
wordtoix = {}
ix = 0
for word in vocab:
    wordtoix[word] = ix
    ixtoword[ix] = word
    ix += 1

# Find the maximum caption length
max_length = max(len(des.split()) for des in all_train_captions)

# Set vocabulary size
vocab_size = len(vocab)

def extract_features(image_path, model):
    # Resize and preprocess image for feature extraction
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    return feature.flatten()

def create_feature_files(image_dir):
    global base_model
    # Load images in correct format (JPG)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        features = extract_features(img_path, base_model)
    return features

# Extract features for training images
image_features = create_feature_files(image_dir)

def load_embedding_matrix(wordtoix, glove_file='glove.6B.200d.txt'):
    # Load GloVe embeddings into a dictionary
    embeddings_index = {}
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word, coefs = values[0], np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    vocab_size = len(wordtoix) + 1
    # Create embedding matrix for training vocabulary
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in wordtoix.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector
    return embedding_matrix

# Load GloVe embedding matrix
embedding_matrix = load_embedding_matrix(wordtoix)

# Split data into training (90%) and validation (10%)
items = list(train_descriptions.items())
split = int(len(items) * 0.9)
train_items = dict(items[:split])
val_items = dict(items[split:])

# Create data generators for training and validation
train_gen = DataGenerator(image_features, train_items, wordtoix, max_length, BATCH_SIZE)
val_gen = DataGenerator(image_features, val_items, wordtoix, max_length, BATCH_SIZE)

def create_model(vocab_size, max_length, embedding_matrix):
    # Image input branch
    image_input = Input(shape=(FEATURE_SIZE,), name="image_input")
    fe1 = Dropout(0.5)(image_input)
    fe modeling = Dense(512, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe modeling)

    # Text input branch
    text_input = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], mask_zero=True, trainable=False)(text_input)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)

    # Merge image and text features
    merged = Add()([fe3, se3])

    # Attention mechanism
    dense_attn = TimeDistributed(Dense(256, activation='tanh'))(merged)
    attention_score = TimeDistributed(Dense(1))(dense_attn)
    attention_weights = Softmax(axis=1)(attention_score)
    context = Multiply()([merged, attention_weights])
    context_vector = ReduceSumLayer()(context)

    # Decoder
    decoder1 = Dense(256, activation='relu')(context_vector)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)

    # Build and compile model
    model = Model(inputs=[image_input, text_input], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model

# Create and summarize the model
model = create_model(vocab_size, max_length, embedding_matrix)
model.summary()

# Train the model
callbacks = [
    ModelCheckpoint("best_model_attention.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# === Commented-out code for calculating model complexity ===
# from keras_flops import get_flops
# from sentence_transformers import SentenceTransformer
# from keras.models import load_model
# from Model_with_attention_improved_2 import ReduceSumLayer
#
# model_temp = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# flops = get_flops(model_temp, batch_size=1)
# print("temp:" + flops)
#
# model = load_model('best_model_attention.keras', custom_objects={"ReduceSumLayer": ReduceSumLayer})
# flops = get_flops(model, batch_size=1)
# print("model:" + flops)