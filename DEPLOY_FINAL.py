import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import os
import pickle
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_or_create_features(model, image_dir, feature_file, filepath_file):
    """Loads or creates feature vectors for the images in a directory."""
    if os.path.exists(feature_file) and os.path.exists(filepath_file):
        print("Loading precomputed features and file paths...")
        feature_vectors = np.load(feature_file, allow_pickle=True)
        filepaths = pickle.load(open(filepath_file, 'rb'))
    else:
        print("Computing features for the images...")
        filepaths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
        feature_vectors = []

        for filepath in tqdm(filepaths, desc="Processing images"):
            features = extract_features(filepath, model)
            feature_vectors.append(features)

        feature_vectors = np.array(feature_vectors)

        # Save the features and file paths
        np.save(feature_file, feature_vectors)
        pickle.dump(filepaths, open(filepath_file, 'wb'))

    return feature_vectors, filepaths


def extract_features(img_path, model):
    """Extracts normalized feature vector from an image using a model."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0).flatten()
    normalized_features = features / norm(features)
    return normalized_features


def display_similar_images(query_img_path, indices, filepaths):
    """Displays the query image and its closest matches."""
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    plt.show()

    # Display similar images
    num_rows, num_cols = 2, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.ravel()

    for i, idx in enumerate(indices[0][1:7]):  # Skip the first match (query image itself)
        similar_img = cv2.imread(filepaths[idx])
        similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(similar_img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Load model
    model = ResNet50(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))
    model.trainable = False

    # Directories and files
    image_dir = input(r'INPUT PATH PRE-PROCESSING = ').strip('"')
    # image_dir = 'C:\Perkuliahan\RUANG KODE\dist\image compressed\images'
    feature_file = 'feature_vector.npy'
    filepath_file = 'filepath.pkl'
    filepaths = pickle.load(open('filepath.pkl', 'rb'))
    print("Number of file paths:", len(filepaths))

    # Load or compute features
    feature_vectors, filepaths = load_or_create_features(model, image_dir, feature_file, filepath_file)

    # Fit Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
    nn_model.fit(feature_vectors)

    # Query image path
    # query_img_path = input('C://Users//sakur//Downloads//kemeja hitam.jpg')
    while True:
        pilih_menu = input(f"-----Select menu-----\n1. Find the similiar cloth\n2. Stop file\nYour choice = ")
        if pilih_menu == "1":
            query_img_path = input(r'INPUT PATH CLOTCH = ').strip('"')
            query_features = extract_features(query_img_path, model)
            distances, indices = nn_model.kneighbors([query_features])

            # Display results
            display_similar_images(query_img_path, indices, filepaths)
        else:
            exit()
    main()


if __name__ == "__main__":
    main()
