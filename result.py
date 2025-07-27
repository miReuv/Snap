import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from Model_with_attention_improved_2 import ReduceSumLayer
import keras

# model and files paths
model_path = './best_model_attention.keras'
wordtoix_path = './wordtoix.pkl'
ixtoword_path = './ixtoword.pkl'
max_length_path = './max_length.txt'

# loading model and files
model = load_model('best_model_attention.keras', custom_objects={"ReduceSumLayer": ReduceSumLayer})

with open(wordtoix_path, 'rb') as f:
    wordtoix = pickle.load(f)
with open(ixtoword_path, 'rb') as f:
    ixtoword = pickle.load(f)
with open(max_length_path, 'r') as f:
    max_length = int(f.read())

def extract_features_from_image(img_path):
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model_cnn = Model(base_model.input, base_model.output)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_for_model = preprocess_input(img.reshape((1, 299, 299, 3)))

    features = model_cnn.predict(img_for_model, verbose=0)
    return features.reshape(2048, ), img  # מחזיר גם את התמונה עצמה


def generate_caption(photo_features):
    in_text = 'startseq'

    for _ in range(max_length):
        sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo_features.reshape(1, 2048), sequence], verbose=0)
        yhat_idx = np.argmax(yhat)
        word = ixtoword.get(yhat_idx)

        if word is None or word == 'endseq':
            break
        in_text += ' ' + word

    return in_text.replace('startseq', '').strip()


test_image_path = './my_image.jpg'
features, img = extract_features_from_image(test_image_path)
caption = generate_caption(features)

# show result
plt.imshow(img)
plt.axis('off')
plt.title(caption, fontsize=12)
plt.show()
