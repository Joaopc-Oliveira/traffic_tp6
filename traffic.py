import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


import os
import cv2
import numpy as np

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43


def load_data(data_dir):
    """
    Load image data and labels from the dataset directory.
    """
    images = []
    labels = []

    # Itera sobre cada diretório de categoria
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        # Verifica se o diretório existe
        if not os.path.isdir(category_path):
            continue

        # Carrega todas as imagens no diretório da categoria
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            # Lê a imagem usando o OpenCV
            img = cv2.imread(file_path)
            if img is None:
                continue

            # Redimensiona a imagem para as dimensões desejadas
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Adiciona a imagem e o rótulo correspondente
            images.append(img)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
