# feature extractor.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
import os

# Load ResNet50 moddel, exclude the top layer andd use global average pooling
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Freeze the model so it's not trainable
base_model.trainable = False

# Loads and preprocesses an image for ResNet50
def load_and_preprocess_image(img_path):
    # Load and resize image to 224x224
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)

    # Convert to array and add batch dimension
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess for ResNet
    img_array = preprocess_input(img_array)
    return img_array

# Extracts feature vector from an image using ResNet50
def extract_features(img_path):
    preprocessed_img = load_and_preprocess_image(img_path)
    features = base_model.predict(preprocessed_img)
    return features.flatten()

# Extract features for a batch of image paths
def batch_extract_features(image_paths):
    feature_list = []
    for path in image_paths:
        features = extract_features(path)
        feature_list.append(features)
    return np.array(feature_list)

if __name__ == '__main__':
    img_file = "dataset/bento-front.jpg"
    features = extract_features(img_file)
    print("Feature vector shape:", features.shape)
    print("First 5 feature values:", features[:5])