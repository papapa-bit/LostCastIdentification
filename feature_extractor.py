# feature_extractor.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load ResNet50 as feature vector
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False

def extract_embedding(img_path):
    # Loads an image, preprocess it, and returns its embedding from ResNet50
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    embedding = base_model.predict(img_array, verbose=0)
    return embedding.flatten()

def extract_embeddings_from_folder(folder_path):
    """
    Extracts embeddings for all images in the folder
    Returns:
        - embeddings: list of 1D numpay arrays
        - image_names: list of filenames
    """
    embeddings = []
    image_names = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(folder_path, fname)
            embedding = extract_embedding(full_path)
            embeddings.append(embedding)
            image_names.append(fname)

    return np.array(embeddings), image_names

if __name__ == '__main__':
    folder = "found_cats"
    embeddings, filenames = extract_embeddings_from_folder(folder)
    np.save("found_embeddings.npy", embeddings)
    np.save("found_filenames.npy", filenames)
    print(f"Saved {len(embeddings)} embeddings.")